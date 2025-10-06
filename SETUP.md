# Environment Setup

## Installation

```bash
# Create conda environment
conda create -n zeroshot-pgt python=3.11 -y
conda activate zeroshot-pgt

# Install all packages with exact versions
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

## Key Package Versions

- **Python**: 3.11
- **PyTorch**: 2.7.1+cu128
- **vLLM**: 0.10.1
- **Flash-Attention**: 2.8.0.post2
- **xformers**: 0.0.31
- **Transformers**: 4.56.2
- **CUDA**: 12.8

## Notes

- Flash-Attention must be version `>=2.7.1,<=2.8.0` for vLLM 0.10.1 compatibility
- vLLM 0.10.1 requires exactly torch 2.7.1 (not 2.8.0)
- FlashInfer is optional and not available for this configuration
- Use `--no-build-isolation` when installing flash-attn to avoid conflicts