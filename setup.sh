#!/bin/bash

echo "=== Ensuring System Dependencies are Installed ==="
apt-get update && apt-get install -y wget ca-certificates

echo "=== Starting LTX-2.5 Template Setup ==="

# Define the correct ComfyUI path for runpod/comfyui
COMFYUI_PATH="/workspace/runpod-slim/ComfyUI"

# The workflows need ComfyUI >= 0.35.0 (LTXVDualCFGGuider does not exist in 0.30.0).
# Use an image that already ships it - this script deliberately does NOT update
# ComfyUI core. Running `pip install -r ComfyUI/requirements.txt` on these images
# can reinstall torch (it is unpinned there) and break the baked CUDA build.
#   Recommended image: runpod/comfyui:1.4.0-rc.164-comfyuiv0.35.0-cuda12.8

# The official weights repo (Lightricks/LTX-2.5) is license-gated. Set HF_TOKEN to a
# read token from an account that accepted the licence. Without it the script falls
# back to an ungated mirror of the same files.
HF_TOKEN="${HF_TOKEN:-}"
[ -n "$HF_TOKEN" ] && export HF_TOKEN
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
OFFICIAL_REPO="Lightricks/LTX-2.5"
MIRROR_REPO="lxxxy6/LTX-2.5"

DOWNLOAD_PROMPT_ENHANCER="${DOWNLOAD_PROMPT_ENHANCER:-1}"
DOWNLOAD_TEMPORAL_UPSCALER="${DOWNLOAD_TEMPORAL_UPSCALER:-1}"

# Self-healing check prevents directory collisions and fixes broken folders
if [ ! -f "$COMFYUI_PATH/main.py" ]; then
  echo "First time setup: Copying baked ComfyUI to workspace..."
  rm -rf "$COMFYUI_PATH"
  mkdir -p /workspace/runpod-slim
  cp -r /opt/comfyui-baked "$COMFYUI_PATH"
fi

if [ ! -f "$COMFYUI_PATH/main.py" ]; then
  echo "FATAL: no ComfyUI at $COMFYUI_PATH and /opt/comfyui-baked did not provide one."
  echo "       Check the container image."
  exit 1
fi

# 1. Move custom nodes to the official ComfyUI directory
# The bundled workflows use core ComfyUI nodes only, so normally there is nothing
# here - this stays for parity with the other templates.
echo "Installing custom nodes..."
mkdir -p "$COMFYUI_PATH/custom_nodes"
if [ -d /tmp/temp_repo/custom_nodes ]; then
  cp -r /tmp/temp_repo/custom_nodes/* "$COMFYUI_PATH/custom_nodes/"
fi

# 2. Automatically find and install requirements for your custom nodes
echo "Installing node requirements..."
find "$COMFYUI_PATH/custom_nodes/" -name "requirements.txt" -exec pip install -r {} \;

# 3. Ensure the correct ComfyUI model folders exist for LTX-2.5
echo "Preparing model directories..."
mkdir -p "$COMFYUI_PATH/models/text_encoders"
mkdir -p "$COMFYUI_PATH/models/diffusion_models"
mkdir -p "$COMFYUI_PATH/models/vae"
mkdir -p "$COMFYUI_PATH/models/loras"
mkdir -p "$COMFYUI_PATH/models/latent_upscale_models"
mkdir -p "$COMFYUI_PATH/models/model_patches"

# -------------------------------------------------------------------
# Download helpers
# -------------------------------------------------------------------
# hf_transfer is a standalone CLI used only to fetch files, so installing it with the
# system pip is fine - nothing in ComfyUI has to import it. It is multi-threaded and
# much faster than wget on 20GB files, but it cannot resume, so wget stays as the
# fallback for flaky connections.
echo "Installing fast downloader (optional)..."
pip install -q -U "huggingface_hub[hf_transfer]" 2>/dev/null
HF_BIN="$(command -v hf || command -v huggingface-cli || true)"
[ -n "$HF_BIN" ] && echo "Fast downloader: $HF_BIN" || echo "Fast downloader unavailable, using wget."

# A truncated HTML error page saved as .safetensors looks plausible until load time,
# so treat anything under 1MB as a failed download.
file_ok() { [ -f "$1" ] && [ "$(stat -c%s "$1")" -ge 1000000 ]; }

hf_fast() {
  # $1 = destination file, $2 = repo id, $3 = path inside the repo
  local dest="$1" repo="$2" rpath="$3" tmp
  [ -n "$HF_BIN" ] || return 1
  # Stage beside the destination so the final move is a rename, not a 20GB copy.
  tmp="$(mktemp -d "$(dirname "$dest")/.hfdl.XXXXXX")" || return 1
  HF_HUB_ENABLE_HF_TRANSFER=1 "$HF_BIN" download "$repo" "$rpath" --local-dir "$tmp" || true
  file_ok "$tmp/$rpath" && mv -f "$tmp/$rpath" "$dest"
  rm -rf "$tmp"
  file_ok "$dest"
}

fetch_hf() {
  # $1 = destination file, $2 = repo id, $3 = path inside the repo
  local dest="$1" repo="$2" rpath="$3"
  local url="https://huggingface.co/$repo/resolve/main/$rpath"
  # Resume into a partial file keyed to the source repo: resuming one repo's partial
  # download against another repo's URL would splice two files together.
  local part="$dest.$(echo "$repo" | tr '/' '_').part"
  if [ -n "$HF_TOKEN" ]; then
    wget -c -q --show-progress --tries=3 --read-timeout=120 \
      --header="Authorization: Bearer $HF_TOKEN" -O "$part" "$url" || true
  fi
  if ! file_ok "$part"; then
    wget -c -q --show-progress --tries=3 --read-timeout=120 -O "$part" "$url" || true
  fi
  file_ok "$part" && mv -f "$part" "$dest"
}

download_ltx_model() {
  # $1 = destination file, $2 = official repo subpath, $3 = bare filename (mirror is flat)
  local dest="$1" sub="$2" fname="$3"
  if [ -f "$dest" ]; then
    echo "$fname already exists, skipping."
    return 0
  fi
  echo "Downloading $fname ..."
  hf_fast "$dest" "$OFFICIAL_REPO" "$sub"   && { echo "$fname done."; return 0; }
  hf_fast "$dest" "$MIRROR_REPO"   "$fname" && { echo "$fname done."; return 0; }
  echo "  falling back to wget..."
  fetch_hf "$dest" "$OFFICIAL_REPO" "$sub"
  file_ok "$dest" || fetch_hf "$dest" "$MIRROR_REPO" "$fname"
  if ! file_ok "$dest"; then
    rm -f "$dest"
    echo "ERROR: could not download $fname."
    return 1
  fi
  echo "$fname done."
}

# -------------------------------------------------------------------
# 4. Model downloads
# -------------------------------------------------------------------

# Diffusion model (22B distilled, int8-convrot, ~21.5GB)
download_ltx_model \
  "$COMFYUI_PATH/models/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" \
  "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" \
  "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"

# Text encoder (Gemma 4 12B with LTX projection, ~15.4GB)
download_ltx_model \
  "$COMFYUI_PATH/models/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors" \
  "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors" \
  "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"

# Video VAE (~1.5GB)
download_ltx_model \
  "$COMFYUI_PATH/models/vae/ltx-2.5-video-vae-bf16.safetensors" \
  "vae/ltx-2.5-video-vae-bf16.safetensors" \
  "ltx-2.5-video-vae-bf16.safetensors"

# Audio VAE (~365MB) - also encodes your input audio in the lipsync workflow
download_ltx_model \
  "$COMFYUI_PATH/models/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  "vae/ltx-2.5-audio-vae-bf16.safetensors" \
  "ltx-2.5-audio-vae-bf16.safetensors"

# Latent spatial upscaler x2 (~260MB) - second stage of both workflows
download_ltx_model \
  "$COMFYUI_PATH/models/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
  "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
  "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

# Auto-duration head (~4MB)
download_ltx_model \
  "$COMFYUI_PATH/models/model_patches/ltx-2.5-duration-head-bf16.safetensors" \
  "model_patches/ltx-2.5-duration-head-bf16.safetensors" \
  "ltx-2.5-duration-head-bf16.safetensors"

# OPTIONAL: latent temporal upscaler x2 (~260MB)
if [ "$DOWNLOAD_TEMPORAL_UPSCALER" = "1" ]; then
  download_ltx_model \
    "$COMFYUI_PATH/models/latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors" \
    "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors" \
    "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"
fi

# OPTIONAL: prompt enhancer text encoder (~8.1GB, its own ungated repo).
# Both workflows carry a CLIPLoader pointing at this file, so keep it unless you
# also edit the workflows.
if [ "$DOWNLOAD_PROMPT_ENHANCER" = "1" ]; then
  PE_DEST="$COMFYUI_PATH/models/text_encoders/gemma4_e2b_it_int8_convrot.safetensors"
  if [ -f "$PE_DEST" ]; then
    echo "gemma4_e2b_it_int8_convrot.safetensors already exists, skipping."
  else
    echo "Downloading prompt enhancer text encoder..."
    hf_fast "$PE_DEST" "Comfy-Org/gemma-4" "text_encoders/gemma4_e2b_it_int8_convrot.safetensors" || \
      fetch_hf "$PE_DEST" "Comfy-Org/gemma-4" "text_encoders/gemma4_e2b_it_int8_convrot.safetensors"
    file_ok "$PE_DEST" && echo "prompt enhancer done." || \
      { rm -f "$PE_DEST"; echo "WARN: prompt enhancer unavailable."; }
  fi
fi

# 5. Install bundled workflows into ComfyUI's Workflows sidebar
echo "Installing workflows..."
WF_DEST="$COMFYUI_PATH/user/default/workflows"
mkdir -p "$WF_DEST"
if [ -d /tmp/temp_repo/workflows ]; then
  # cp -n: never overwrite a workflow already edited on this pod
  cp -n /tmp/temp_repo/workflows/*.json "$WF_DEST/" 2>/dev/null
  ls "$WF_DEST" | sed 's/^/  workflow: /'
else
  echo "No workflows directory found in repo, skipping."
fi

# 6. Clean up the temporary git folder
echo "Cleaning up temp files..."
rm -rf /tmp/temp_repo

# 7. Start ComfyUI using the official RunPod entrypoint
echo "Setup complete! Handing over to start script..."
exec /start.sh
