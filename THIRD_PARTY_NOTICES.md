# Third-party components

Project-owned preprocessing, wavelet/fusion modules, task heads, training orchestration, and tests are covered by the repository's MIT license. No NVIDIA backbone implementation or pretrained weights are vendored here.

- **MambaVision / MambaVision-T-1K**: loaded separately using Transformers. See the [official repository](https://github.com/NVlabs/MambaVision), [model card](https://huggingface.co/nvidia/MambaVision-T-1K), and [NVIDIA Source Code License-NC](https://github.com/NVlabs/MambaVision/blob/main/LICENSE). Upstream limits the covered work to non-commercial research/evaluation. The project MIT license does not relicense the backbone or weights.
- **mamba-ssm**: the CUDA/runtime dependency is obtained separately; see its [official source and license](https://github.com/state-spaces/mamba).
- PyTorch, torchvision, Transformers, timm, einops, numpy, pandas, scikit-learn, Pillow, tqdm, and pytest are installed dependencies. Their respective licenses remain applicable.

When redistributing upstream code or weights, preserve the applicable upstream license and notices. Pinning `--model_revision` identifies the precise downloaded backbone revision.
