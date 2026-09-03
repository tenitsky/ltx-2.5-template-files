#!/bin/bash

echo "=== Ensuring System Dependencies are Installed ==="
apt-get update && apt-get install -y wget ca-certificates git

echo "=== Starting LTX-2.5 Template Setup ==="

# -------------------------------------------------------------------
# Configuration (all overridable via RunPod template environment vars)
# -------------------------------------------------------------------
COMFYUI_PATH="/workspace/runpod-slim/ComfyUI"

# Update ComfyUI core + UI packages so the native LTX-2.5 nodes and the
# built-in LTX-2.5 T2V / I2V / FLF2V workflow templates are available.
UPDATE_COMFYUI="${UPDATE_COMFYUI:-1}"

# Optional extras (8GB prompt-enhancer encoder, latent temporal upscaler)
DOWNLOAD_PROMPT_ENHANCER="${DOWNLOAD_PROMPT_ENHANCER:-1}"
DOWNLOAD_TEMPORAL_UPSCALER="${DOWNLOAD_TEMPORAL_UPSCALER:-1}"

# The official weights repo (Lightricks/LTX-2.5) is license-gated: set HF_TOKEN
# to a Hugging Face read token whose account has accepted the LTX-2.x license.
# If no token is set, or the gated download fails, the script automatically
# falls back to an ungated mirror of the exact same files (lxxxy6/LTX-2.5).
HF_TOKEN="${HF_TOKEN:-}"
OFFICIAL_REPO="Lightricks/LTX-2.5"
MIRROR_REPO="lxxxy6/LTX-2.5"
# Official ungated source for the prompt-enhancer encoder (Comfy-Org/gemma-4);
# personal-mirror fallback kept in case the official file moves.
ENHANCER_REPO="Comfy-Org/gemma-4"
ENHANCER_REPO_FALLBACK="patientxtr/gemma-4-E2B-it-int8-convrot"

# Repo root (works whether the template repo is cloned to /tmp/temp_repo or anywhere else)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
file_ok() { [ -f "$1" ] && [ "$(stat -c%s "$1")" -ge 1000000 ]; }

fetch_hf() {
  # $1 = destination file, $2 = repo id, $3 = path inside the repo
  local dest="$1" repo="$2" rpath="$3"
  local url="https://huggingface.co/$repo/resolve/main/$rpath"
  if [ -n "$HF_TOKEN" ]; then
    wget -c -q --show-progress --tries=3 --read-timeout=120 \
      --header="Authorization: Bearer $HF_TOKEN" -O "$dest" "$url" || true
  fi
  if ! file_ok "$dest"; then
    wget -c -q --show-progress --tries=3 --read-timeout=120 -O "$dest" "$url" || true
  fi
}

download_ltx_model() {
  # $1 = destination file, $2 = official repo subpath, $3 = bare filename
  local dest="$1" sub="$2" fname="$3"
  if [ -f "$dest" ]; then
    echo "$fname already exists, skipping."
    return 0
  fi
  echo "Downloading $fname ..."
  mkdir -p "$(dirname "$dest")"
  fetch_hf "$dest" "$OFFICIAL_REPO" "$sub"
  if ! file_ok "$dest"; then
    echo "Gated download failed for $fname (missing HF_TOKEN?), trying ungated mirror..."
    fetch_hf "$dest" "$MIRROR_REPO" "$fname"
  fi
  if ! file_ok "$dest"; then
    rm -f "$dest"
    echo "ERROR: could not download $fname from $OFFICIAL_REPO or $MIRROR_REPO."
    return 1
  fi
  echo "$fname done ($(du -h "$dest" | cut -f1))."
}

# -------------------------------------------------------------------
# ComfyUI
# -------------------------------------------------------------------
# Define the correct ComfyUI path for runpod/comfyui:cuda12.8

# If ComfyUI is not yet in the workspace (or missing main.py), copy the pre-built files first
# This self-healing check prevents directory collisions and fixes broken folders
if [ ! -f "$COMFYUI_PATH/main.py" ]; then
  echo "First time setup: Copying baked ComfyUI to workspace..."
  # Clean up any broken, empty directory from previous failed setups first
  rm -rf "$COMFYUI_PATH"
  mkdir -p /workspace/runpod-slim
  cp -r /opt/comfyui-baked "$COMFYUI_PATH"
fi

# 0. Update ComfyUI core so LTX-2.5 native nodes + workflow templates exist
if [ "$UPDATE_COMFYUI" = "1" ]; then
  echo "Updating ComfyUI core (needed for native LTX-2.5 support)..."
  if [ -d "$COMFYUI_PATH/.git" ]; then
    git -C "$COMFYUI_PATH" pull --ff-only || echo "WARN: git pull failed, keeping baked version."
  else
    echo "WARN: $COMFYUI_PATH is not a git repo, cannot pull updates."
  fi
  pip install -q -U comfyui-frontend-package comfyui-workflow-templates || \
    echo "WARN: could not update UI/workflow-template packages."
fi

# 1. Move custom nodes to the official ComfyUI directory
echo "Installing custom nodes..."
mkdir -p "$COMFYUI_PATH/custom_nodes"
# Reference-flow compatibility: use pre-cloned repo nodes if the template provided them
if [ -d /tmp/temp_repo/custom_nodes ]; then
  cp -r /tmp/temp_repo/custom_nodes/* "$COMFYUI_PATH/custom_nodes/" 2>/dev/null || true
fi
# Official Lightricks node pack (extra LTX I2V/advanced nodes on top of native support)
if [ ! -d "$COMFYUI_PATH/custom_nodes/ComfyUI-LTXVideo" ]; then
  git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo \
    "$COMFYUI_PATH/custom_nodes/ComfyUI-LTXVideo" || \
    echo "WARN: could not clone ComfyUI-LTXVideo; native LTX-2.5 nodes still work."
fi

# 2. Automatically find and install requirements for your custom nodes
echo "Installing node requirements..."
find "$COMFYUI_PATH/custom_nodes/" -name "requirements.txt" -exec pip install -r {} \;

# 2.5 Install bundled workflows into ComfyUI's workflow menu
echo "Installing bundled workflows..."
mkdir -p "$COMFYUI_PATH/user/default/workflows"
if [ -d "$SCRIPT_DIR/workflows" ]; then
  # cp -n: never overwrite a workflow the user already edited on a previous boot
  find "$SCRIPT_DIR/workflows" -name "*.json" -exec cp -n {} "$COMFYUI_PATH/user/default/workflows/" \;
  ls "$COMFYUI_PATH/user/default/workflows" | sed 's/^/  workflow: /'
else
  echo "No workflows directory found in repo, skipping."
fi

# 3. Ensure the correct ComfyUI model folders exist
echo "Preparing model directories..."
mkdir -p "$COMFYUI_PATH/models/text_encoders"
mkdir -p "$COMFYUI_PATH/models/diffusion_models"
mkdir -p "$COMFYUI_PATH/models/vae"
mkdir -p "$COMFYUI_PATH/models/loras"
mkdir -p "$COMFYUI_PATH/models/latent_upscale_models"
mkdir -p "$COMFYUI_PATH/models/model_patches"

# -------------------------------------------------------------------
# 4. Model downloads (LTX-2.5, per ComfyUI official requirements)
# -------------------------------------------------------------------

# 4.1 Diffusion Model (LTX-2.5 22B distilled, int8-convrot, ~21.5 GB)
download_ltx_model \
  "$COMFYUI_PATH/models/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" \
  "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" \
  "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" || exit 1

# 4.2 Text Encoder (Gemma 4 12B with LTX projection, int8-convrot, ~15.4 GB)
download_ltx_model \
  "$COMFYUI_PATH/models/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors" \
  "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors" \
  "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors" || exit 1

# 4.3 Video VAE (~1.5 GB)
download_ltx_model \
  "$COMFYUI_PATH/models/vae/ltx-2.5-video-vae-bf16.safetensors" \
  "vae/ltx-2.5-video-vae-bf16.safetensors" \
  "ltx-2.5-video-vae-bf16.safetensors" || exit 1

# 4.4 Audio VAE (~365 MB, synced audio generation)
download_ltx_model \
  "$COMFYUI_PATH/models/vae/ltx-2.5-audio-vae-bf16.safetensors" \
  "vae/ltx-2.5-audio-vae-bf16.safetensors" \
  "ltx-2.5-audio-vae-bf16.safetensors" || exit 1

# 4.5 Latent Spatial Upscaler x2 (~260 MB, hi-res I2V workflow)
download_ltx_model \
  "$COMFYUI_PATH/models/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
  "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
  "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" || exit 1

# 4.6 Auto-duration head (~4 MB model patch)
download_ltx_model \
  "$COMFYUI_PATH/models/model_patches/ltx-2.5-duration-head-bf16.safetensors" \
  "model_patches/ltx-2.5-duration-head-bf16.safetensors" \
  "ltx-2.5-duration-head-bf16.safetensors" || exit 1

# 4.7 OPTIONAL: Latent Temporal Upscaler x2 (~260 MB)
if [ "$DOWNLOAD_TEMPORAL_UPSCALER" = "1" ]; then
  download_ltx_model \
    "$COMFYUI_PATH/models/latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors" \
    "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors" \
    "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors" || \
    echo "WARN: temporal upscaler missing, workflows without it still run."
fi

# 4.8 OPTIONAL: Prompt Enhancer text encoder (gemma-4-E2B int8, ~8.1 GB).
# Not in the Lightricks repo - lives in its own ungated repo.
if [ "$DOWNLOAD_PROMPT_ENHANCER" = "1" ]; then
  PE_DEST="$COMFYUI_PATH/models/text_encoders/gemma4_e2b_it_int8_convrot.safetensors"
  if [ -f "$PE_DEST" ]; then
    echo "gemma4_e2b_it_int8_convrot.safetensors already exists, skipping."
  else
    echo "Downloading prompt enhancer text encoder (gemma4_e2b_it_int8_convrot)..."
    fetch_hf "$PE_DEST" "$ENHANCER_REPO" "text_encoders/gemma4_e2b_it_int8_convrot.safetensors"
    if ! file_ok "$PE_DEST"; then
      echo "Official source failed, trying fallback mirror..."
      fetch_hf "$PE_DEST" "$ENHANCER_REPO_FALLBACK" "gemma4_e2b_it_int8_convrot.safetensors"
    fi
    if file_ok "$PE_DEST"; then
      echo "gemma4_e2b_it_int8_convrot.safetensors done."
    else
      rm -f "$PE_DEST"
      echo "WARN: prompt enhancer encoder unavailable; only the optional prompt enhancer is affected."
    fi
  fi
fi

# 5. Clean up the temporary git folder
echo "Cleaning up temp files..."
rm -rf /tmp/temp_repo

# 6. Start ComfyUI using the official RunPod entrypoint
echo "Setup complete! Handing over to start script..."
exec /start.sh