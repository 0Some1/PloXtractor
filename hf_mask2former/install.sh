#!/bin/bash
# install.sh
set -e

echo ">>> Checking for uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

echo ">>> Creating environment..."
uv venv .venv --python 3.10
source .venv/bin/activate

echo ">>> Installing PyTorch..."
# Install Torch compatible with your CUDA version
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

echo ">>> Installing Hugging Face Stack..."
uv pip install -r requirements.txt

echo ">>> Setup Complete. Activate with: source .venv/bin/activate"