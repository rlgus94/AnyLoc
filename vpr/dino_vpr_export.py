"""Export the universal AnyLoc loop-closure head: DINOv2 ViT-S/14 + VLAD.

The loop-closure retrieval descriptor is a SINGLE extractor-independent head:

    image -> DINOv2 ViT-S/14 (frozen) -> dense patch tokens (T x 384)
          -> VLAD against a fixed vocabulary -> L2-normalized global descriptor

Two artifacts are produced, consumed by the C++ ``DinoVpr`` runtime in
``DL-VINS/src/loop_fusion``:

  1. A SINGLE STRICT fixed-resolution TensorRT engine (no dynamic shapes). 
     Input is letterboxed to one fixed ``HxW`` that is a multiple of 14 (ViT-S/14 patch size).
     ONNX is always written; the ``.engine`` build needs a GPU + TensorRT.

  2. A VLAD vocabulary ``.bin`` in the same layout ``VladCodebook::load`` reads:
         int32 clusters | int32 dim(=384) | clusters*dim float32 (row-major)
     fit by k-means over DINO tokens pooled from outside evaluation imagery,
     so evaluation stays honest.

No retraining: the DINOv2 backbone is frozen (weights via ``torch.hub``); the
vocabulary is unsupervised k-means.

Usage
-----
    # ONNX + vocabulary fit (no GPU engine build):
    uv run python -m vpr.dino_vpr_export \\
        --image-roots /data/baidu_mall/images /data/gardens \\
        --input-h 280 --input-w 448 --clusters 32 --onnx-only

    # Full export including the TensorRT engine (needs GPU + tensorrt):
    uv run python -m vpr.dino_vpr_export \\
        --image-roots /data/baidu_mall/images /data/gardens \\
        --input-h 280 --input-w 448 --clusters 32 --fp16
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from vpr import config
from vpr.descriptors import _vlad

# DINOv2 ViT-S/14 embedding dimension.
DINO_VITS14_DIM = 384
DINO_PATCH = 14
# ImageNet normalization (DINOv2 preprocessing). Kept here as the single source
# of truth; the C++ DinoVpr preprocess kernel must use the SAME constants.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_OUT_DIR = config.REPO_ROOT / "DL-VINS" / "src" / "loop_fusion" / "support_files"
_IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def _check_multiple_of_patch(name: str, value: int) -> None:
    if value % DINO_PATCH != 0:
        raise SystemExit(
            f"--{name}={value} must be a multiple of {DINO_PATCH} (ViT-S/14 patch size)"
        )


def _list_images(root: Path) -> list[Path]:
    paths: list[Path] = []
    for pat in _IMAGE_GLOBS:
        paths.extend(root.rglob(pat))
    seen: set[Path] = set()
    out: list[Path] = []
    for p in sorted(paths):
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


class _DinoPatchTokens:
    """Lazy DINOv2 ViT-S/14 wrapper exposing dense patch tokens (B, T, 384)."""

    def __init__(self, device: str):
        import torch

        self._torch = torch
        self._device = device
        # Frozen self-supervised backbone; downloaded + cached by torch.hub.
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self._model = model.to(device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

    @property
    def module(self):
        return self._model

    def tokens(self, x):
        """x: (B, 3, H, W) float32 already ImageNet-normalized -> (B, T, 384)."""
        torch = self._torch
        with torch.no_grad():
            feats = self._model.forward_features(x)
        return feats["x_norm_patchtokens"]  # (B, T, 384)


class _ExportWrapper:
    """nn.Module-like wrapper returning only patch tokens (for clean ONNX export)."""

    def __new__(cls, backbone: _DinoPatchTokens):
        import torch.nn as nn

        class _M(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                feats = self.model.forward_features(x)
                return feats["x_norm_patchtokens"]

        return _M(backbone.module)


def export_onnx(backbone: _DinoPatchTokens, input_h: int, input_w: int,
                onnx_path: Path) -> None:
    import torch

    # NOTE: _ExportWrapper wraps the SAME nn.Module instance as `backbone`, so
    # moving it to CPU for export also moves `backbone`. Snapshot the device and
    # restore it afterwards so the subsequent GPU vocabulary fit still works.
    dev = backbone._device
    wrapper = _ExportWrapper(backbone).to("cpu").eval()
    dummy = torch.zeros(1, 3, input_h, input_w, dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            input_names=["image"],
            output_names=["tokens"],
            opset_version=18,
            dynamic_axes=None,  # SINGLE strict fixed-resolution engine: no dynamic shapes
            do_constant_folding=True,
        )
        print(f"  [onnx] wrote {onnx_path}  (image 1x3x{input_h}x{input_w} -> tokens)")
    finally:
        backbone.module.to(dev)


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool) -> bool:
    """Build a static-shape TensorRT engine from ONNX. Returns False if no TRT."""
    try:
        import tensorrt as trt
    except Exception:
        return _build_engine_trtexec(onnx_path, engine_path, fp16)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i), file=sys.stderr)
            return False
    cfg_trt = builder.create_builder_config()
    cfg_trt.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if fp16 and builder.platform_has_fast_fp16:
        cfg_trt.set_flag(trt.BuilderFlag.FP16)
    serialized = builder.build_serialized_network(network, cfg_trt)
    if serialized is None:
        print("  [engine] build_serialized_network returned None", file=sys.stderr)
        return False
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"  [engine] wrote {engine_path}  (fp16={fp16})")
    return True


def _build_engine_trtexec(onnx_path: Path, engine_path: Path, fp16: bool) -> bool:
    exe = "trtexec"
    cmd = [exe, f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
    if fp16:
        cmd.append("--fp16")
    try:
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)
        print(f"  [engine] trtexec wrote {engine_path}")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"  [engine] skipped (no TensorRT / trtexec): {e}", file=sys.stderr)
        return False


def _letterbox_normalized(gray_or_bgr: np.ndarray, input_h: int,
                          input_w: int) -> np.ndarray:
    """Replicate the C++ DinoVpr preprocess exactly: resize keeping aspect,
    letterbox-pad to (input_h, input_w), gray->RGB, /255, ImageNet mean/std,
    NCHW. Returns (3, H, W) float32.
    """
    h, w = gray_or_bgr.shape[:2]
    scale = min(input_w / w, input_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(gray_or_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    else:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    canvas = np.zeros((input_h, input_w, 3), np.float32)
    pad_x = (input_w - new_w) // 2
    pad_y = (input_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized.astype(np.float32)
    canvas /= 255.0
    canvas = (canvas - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return np.transpose(canvas, (2, 0, 1)).copy()


def fit_vocabulary(backbone: _DinoPatchTokens, image_roots: list[Path],
                   input_h: int, input_w: int, clusters: int,
                   tokens_per_image: int, sample_cap: int, seed: int) -> np.ndarray:
    """Pool DINO patch tokens over held-out imagery and k-means a VLAD vocab.

    Returns (clusters, 384) float32 cluster centres.
    """
    import torch

    rng = np.random.default_rng(seed)
    pool: list[np.ndarray] = []
    total = 0
    for root in image_roots:
        imgs = _list_images(root)
        if not imgs:
            print(f"  [skip] no images under {root}")
            continue
        for p in imgs:
            try:
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None:
                    continue
            except Exception:
                continue
            chw = _letterbox_normalized(img, input_h, input_w)
            x = torch.from_numpy(chw)[None].to(backbone._device)
            tok = backbone.tokens(x)[0].float().cpu().numpy()  # (T, 384)
            take = min(tokens_per_image, len(tok))
            idx = rng.choice(len(tok), take, replace=False)
            pool.append(tok[idx].astype(np.float32))
        total += len(imgs)
        print(f"  [pool] {root.name}: {len(imgs)} imgs")

    if not pool:
        raise SystemExit("no tokens pooled — check --image-roots")
    sample = np.concatenate(pool, axis=0)
    if len(sample) > sample_cap:
        sample = sample[rng.choice(len(sample), sample_cap, replace=False)]
    print(f"  [kmeans] {len(sample)} tokens over {total} images -> {clusters} clusters")
    km = _vlad.build_codebook(sample, clusters, seed)
    return np.ascontiguousarray(km.cluster_centers_, dtype=np.float32)


def write_vocab_bin(centers: np.ndarray, out_path: Path) -> None:
    k, d = centers.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fp:
        fp.write(struct.pack("<ii", int(k), int(d)))
        fp.write(np.ascontiguousarray(centers, np.float32).tobytes())
    print(f"  [vocab] {k}x{d} -> {out_path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-roots", nargs="*", type=Path, default=[],
                   help="Held-out VPR image dirs for vocabulary k-means "
                        "(NOT the EuRoC eval datasets). Omit with --onnx-only "
                        "to skip vocabulary fitting.")
    p.add_argument("--input-h", type=int, default=280,
                   help="Engine input height (multiple of 14). Default 280 (20 patches).")
    p.add_argument("--input-w", type=int, default=448,
                   help="Engine input width (multiple of 14). Default 448 (32 patches).")
    p.add_argument("--clusters", type=int, default=32, help="VLAD k.")
    p.add_argument("--tokens-per-image", type=int, default=256)
    p.add_argument("--sample-cap", type=int, default=400_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fp16", action="store_true", help="Build the engine in FP16.")
    p.add_argument("--onnx-only", action="store_true",
                   help="Export ONNX + (optional) vocabulary; skip engine build.")
    p.add_argument("--engine-out", type=Path, default=None,
                   help="Engine path (default: support_files/dino_vits14_h<H>_w<W>.fp16.engine).")
    p.add_argument("--onnx-out", type=Path, default=None)
    p.add_argument("--vocab-out", type=Path, default=None)
    args = p.parse_args(argv)

    _check_multiple_of_patch("input-h", args.input_h)
    _check_multiple_of_patch("input-w", args.input_w)

    suffix = f"h{args.input_h}_w{args.input_w}"
    onnx_path = args.onnx_out or (_OUT_DIR / f"dino_vits14_{suffix}.onnx")
    prec = "fp16" if args.fp16 else "fp32"
    engine_path = args.engine_out or (_OUT_DIR / f"dino_vits14_{suffix}.{prec}.engine")
    vocab_path = args.vocab_out or (_OUT_DIR / f"vlad_vocab_dino_vits14_k{args.clusters}.bin")

    device = config.torch_device()
    print(f"DINOv2 ViT-S/14 export  device={device}  input={args.input_h}x{args.input_w}  k={args.clusters}")
    backbone = _DinoPatchTokens(device)

    export_onnx(backbone, args.input_h, args.input_w, onnx_path)

    if not args.onnx_only:
        build_engine(onnx_path, engine_path, args.fp16)

    if args.image_roots:
        roots = [r.expanduser().resolve() for r in args.image_roots]
        missing = [r for r in roots if not r.is_dir()]
        if missing:
            print(f"image roots not found: {missing}", file=sys.stderr)
            return 1
        centers = fit_vocabulary(backbone, roots, args.input_h, args.input_w,
                                 args.clusters, args.tokens_per_image,
                                 args.sample_cap, args.seed)
        write_vocab_bin(centers, vocab_path)
    else:
        print("  [vocab] no --image-roots given; skipped vocabulary fit "
              "(drop in an AnyLoc domain vocabulary instead)")

    meta = {
        "backbone": "dinov2_vits14",
        "embed_dim": DINO_VITS14_DIM,
        "patch": DINO_PATCH,
        "input_h": args.input_h,
        "input_w": args.input_w,
        "clusters": args.clusters,
        "vlad_dim": args.clusters * DINO_VITS14_DIM,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "precision": prec,
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "vocab": str(vocab_path),
    }
    meta_path = _OUT_DIR / f"dino_vits14_{suffix}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  [meta] {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
