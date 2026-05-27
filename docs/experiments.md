# Experiments Notes

## Batch Size Differences: ResNet-18 vs ViT-B/32

ResNet-18 uses `batch_size: 256`; ViT-B/32 uses `batch_size: 64` due to GPU memory constraints
(224×224 inputs + two forward passes for SAM-family optimizers exhaust Apple M4 memory at 256).

**This does not invalidate cross-architecture comparisons** because:

1. **Primary comparisons are within-architecture.** SGD vs SAM vs ASAM vs M-SAM are each
   evaluated at the same batch size within their architecture (256 for ResNet-18, 64 for ViT).
   Optimizer rankings within each architecture remain fair.

2. **Cross-architecture comparison is qualitative, not quantitative.** The research question is
   whether reparametrization invariance and sharpness patterns *generalise* from ResNet-18 to
   ViT — e.g. does M-SAM still achieve the lowest tr(H)/d, does ASAM still show a lower
   divergence rate? Absolute accuracy numbers are not compared across architectures.

**Learning rate note:** The linear scaling rule suggests lr ∝ batch_size. ViT already uses
`lr: 0.01` vs ResNet-18's `lr: 0.1` (10× lower vs 4× batch reduction), which is appropriate
since ViT is fine-tuned from pretrained ImageNet weights and warrants a lower LR regardless.

**For the final report:** include a sentence such as:
> "ViT-B/32 experiments use batch size 64 due to GPU memory constraints; all four optimizers
> are evaluated under identical conditions within each architecture."
