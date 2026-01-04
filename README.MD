# 8x8 Digit Recognition Neural Network Accelerator (SystemVerilog)

This project builds a small inference accelerator that classifies 8x8 handwritten digit images (0-9)
using a 2-layer MLP (64 -> 32 -> 10) with ReLU.

## Repo structure
- sw/ : Python scripts to train + export quantized weights and test vectors
- rtl/: SystemVerilog RTL for the accelerator (coming soon)
- tb/ : SystemVerilog testbenches (coming soon)
- data/: exported .mem files loaded by RTL

## Generate model files
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python sw/make_digit_data.py
