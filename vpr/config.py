"""Central configuration for the offline AnyLoc-head exporter.

This package exports the DINOv2 ViT-S/14 + VLAD loop-closure head consumed by
``DL-VINS/src/loop_fusion`` at runtime. The VLAD vocabulary is fit (unsupervised
k-means) over DINO patch tokens pooled from one or more held-out VPR datasets
(see ``vpr/dino_vpr_export.py`` ``--image-roots``); evaluation of the C++
loop-closure path happens against the DL-VINS dataset roots, which are
*never* fed into vocabulary fitting.

Layout:  <repo>/AnyLoc/vpr/config.py
  VPR_PKG_DIR  = <repo>/AnyLoc/vpr
  ANYLOC_ROOT  = <repo>/AnyLoc
  REPO_ROOT    = <repo>              (weights/, tmp/, LightGlue-ONNX-Jetson/)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── repo layout ───────────────────────────────────────────────────────────────
VPR_PKG_DIR: Path = Path(__file__).resolve().parent
ANYLOC_ROOT: Path = VPR_PKG_DIR.parent
REPO_ROOT: Path = ANYLOC_ROOT.parent

# make ``import vpr`` resolve no matter the cwd
if str(ANYLOC_ROOT) not in sys.path:
    sys.path.insert(0, str(ANYLOC_ROOT))

def torch_device() -> str:
    """Resolve the torch device lazily."""
    forced = os.environ.get("VPR_DEVICE", "").strip()
    if forced:
        return forced
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
