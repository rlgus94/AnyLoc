"""Offline exporter for the DL-VINS loop_fusion AnyLoc head.

The C++ ``loop_fusion`` package runs a single extractor-independent global
place-recognition head at runtime: DINOv2 ViT-S/14 + VLAD over DINO patch
tokens. This package exports that head's artifacts (a fixed-resolution
TensorRT engine and a VLAD vocabulary) from one or more held-out VPR image
roots — never the DL-VINS evaluation datasets, to keep evaluation honest.

Entry point: ``python -m vpr.dino_vpr_export --image-roots ... [--fp16]``
"""

__all__ = ["config"]
