#!/bin/bash

echo "=== Ensuring System Dependencies are Installed ==="
apt-get update && apt-get install -y wget ca-certificates

echo "=== Starting Stable Diffusion 3 Template Setup ==="

# Define the correct ComfyUI path for runpod/comfyui:cuda12.8
COMFYUI_PATH="/workspace/runpod-slim/ComfyUI"

# Self-healing check prevents directory collisions and fixes broken folders
if [ ! -f "$COMFYUI_PATH/main.py" ]; then
  echo "First time setup: Copying baked ComfyUI to workspace..."
  rm -rf "$COMFYUI_PATH"
  mkdir -p /workspace/runpod-slim
  cp -r /opt/comfyui-baked "$COMFYUI_PATH"
fi

# 1. Move custom nodes to the official ComfyUI directory
echo "Installing custom nodes..."
mkdir -p "$COMFYUI_PATH/custom_nodes"
if [ -d /tmp/temp_repo/custom_nodes ]; then
  cp -r /tmp/temp_repo/custom_nodes/* "$COMFYUI_PATH/custom_nodes/"
fi

# 2. Automatically find and install requirements for your custom nodes
echo "Installing node requirements..."
find "$COMFYUI_PATH/custom_nodes/" -name "requirements.txt" -exec pip install -r {} \;

# 3. Ensure the correct ComfyUI model folders exist for SD3
echo "Preparing model directories..."
mkdir -p "$COMFYUI_PATH/models/text_encoders"
mkdir -p "$COMFYUI_PATH/models/diffusion_models"
mkdir -p "$COMFYUI_PATH/models/vae"
mkdir -p "$COMFYUI_PATH/models/loras"

# 4. Check and download SD3 Text Encoders (CLIP-L & OpenCLIP-G bundled, plus full T5-XXL)
# Note: T5-XXL FP16 is ~9.5GB and critical for prompt adherence on high-end hardware
if [ ! -f "$COMFYUI_PATH/models/text_encoders/t5xxl_fp16.safetensors" ]; then
  echo "Downloading Text Encoder (T5-XXL FP16)..."
  wget -q --show-progress -O "$COMFYUI_PATH/models/text_encoders/t5xxl_fp16.safetensors" "https://huggingface.co"
else
  echo "T5 Text Encoder already exists, skipping."
fi

if [ ! -f "$COMFYUI_PATH/models/text_encoders/clip_l.safetensors" ]; then
  echo "Downloading Text Encoder (CLIP-L)..."
  wget -q --show-progress -O "$COMFYUI_PATH/models/text_encoders/clip_l.safetensors" "https://huggingface.co"
else
  echo "CLIP-L Text Encoder already exists, skipping."
fi

# 5. Check and download Diffusion Model (SD3 Medium 2B Base MMDiT)
if [ ! -f "$COMFYUI_PATH/models/diffusion_models/sd3_medium.safetensors" ]; then
  echo "Downloading SD3 Medium Base Model..."
  wget -q --show-progress -O "$COMFYUI_PATH/models/diffusion_models/sd3_medium.safetensors" "https://huggingface.co"
else
  echo "SD3 Diffusion Model already exists, skipping."
fi

# 6. Check and download SD3 Specific VAE
if [ ! -f "$COMFYUI_PATH/models/vae/sd3_vae.safetensors" ]; then
  echo "Downloading SD3 VAE..."
  wget -q --show-progress -O "$COMFYUI_PATH/models/vae/sd3_vae.safetensors" "https://huggingface.co"
else
  echo "SD3 VAE already exists, skipping."
fi

# 7. Install LoRA files from your repo if present
echo "Installing LoRA files..."
if [ -d /tmp/temp_repo/lora ]; then
  for f in /tmp/temp_repo/lora/*.safetensors; do
    [ -e "$f" ] || continue
    fname=$(basename "$f")
    dest="$COMFYUI_PATH/models/loras/$fname"
    if [ -f "$dest" ]; then
      echo "LoRA $fname already exists, skipping."
      continue
    fi
    cp "$f" "$dest"
  done
else
  echo "No lora directory found in repo, skipping."
fi

# 8. Clean up the temporary git folder
echo "Cleaning up temp files..."
rm -rf /tmp/temp_repo

# 9. Start ComfyUI using the official RunPod entrypoint
echo "Setup complete! Handing over to start script..."
exec /start.sh
