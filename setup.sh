#!/bin/bash

echo "=== Ensuring System Dependencies are Installed ==="
apt-get update && apt-get install -y wget ca-certificates git

echo "=== Starting LTX-2.5 Template Setup ==="

# -------------------------------------------------------------------
# Configuration (all overridable via RunPod template environment vars)
# -------------------------------------------------------------------
# Where ComfyUI lives. Overridable as a template env var if your image differs.
COMFYUI_PATH="${COMFYUI_PATH:-/workspace/runpod-slim/ComfyUI}"

# Update ComfyUI core + UI packages so the native LTX-2.5 nodes and the
# built-in LTX-2.5 T2V / I2V / FLF2V workflow templates are available.
UPDATE_COMFYUI="${UPDATE_COMFYUI:-1}"

# Optional extras (8GB prompt-enhancer encoder, latent temporal upscaler)
DOWNLOAD_PROMPT_ENHANCER="${DOWNLOAD_PROMPT_ENHANCER:-1}"
DOWNLOAD_TEMPORAL_UPSCALER="${DOWNLOAD_TEMPORAL_UPSCALER:-1}"

# Every workflow this template ships uses core ComfyUI nodes only, so the Lightricks
# node pack is off by default: its requirements (openimageio, diffusers, a transformers
# bump) often fail to build and can disturb the baked environment. Set to 1 if you want
# the extra LTXV nodes for your own workflows.
INSTALL_LTXVIDEO_NODES="${INSTALL_LTXVIDEO_NODES:-0}"

# The official weights repo (Lightricks/LTX-2.5) is license-gated: set HF_TOKEN
# to a Hugging Face read token whose account has accepted the LTX-2.x license.
# If no token is set, or the gated download fails, the script automatically
# falls back to an ungated mirror of the exact same files (lxxxy6/LTX-2.5).
HF_TOKEN="${HF_TOKEN:-}"
# huggingface_hub reads HF_TOKEN from the environment; export it so the fast
# downloader authenticates too (wget passes it as a header separately).
[ -n "$HF_TOKEN" ] && export HF_TOKEN
# Keep any HF cache on the volume, never on the 5GB container disk.
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
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
  # Resume into a partial file keyed to the source repo, so an interrupted download
  # from one repo is never resumed against a different repo's URL (which would splice
  # two files together and silently corrupt the weights).
  local part="$dest.$(echo "$repo" | tr '/' '_').part"
  if [ -n "$HF_TOKEN" ]; then
    wget -c -q --show-progress --tries=3 --read-timeout=120 \
      --header="Authorization: Bearer $HF_TOKEN" -O "$part" "$url" || true
  fi
  if ! file_ok "$part"; then
    wget -c -q --show-progress --tries=3 --read-timeout=120 -O "$part" "$url" || true
  fi
  if file_ok "$part"; then
    mv -f "$part" "$dest"
  fi
}

hf_fast() {
  # $1 = destination file, $2 = repo id, $3 = path inside the repo
  # hf_transfer opens many parallel connections - typically several times faster than
  # wget's single stream on a 21GB file. It cannot resume, so wget remains the fallback.
  local dest="$1" repo="$2" rpath="$3"
  [ -n "$HF_BIN" ] && [ -x "$HF_BIN" ] || return 1
  local tmp
  tmp="$(mktemp -d "$(dirname "$dest")/.hfdl.XXXXXX")" || return 1
  HF_HUB_ENABLE_HF_TRANSFER=1 "$HF_BIN" download "$repo" "$rpath" --local-dir "$tmp" || true
  if file_ok "$tmp/$rpath"; then
    mv -f "$tmp/$rpath" "$dest"
  fi
  rm -rf "$tmp"
  file_ok "$dest"
}

download_ltx_model() {
  # $1 = destination file, $2 = official repo subpath, $3 = bare filename
  # Four attempts, fastest first: hf official -> hf mirror -> wget official -> wget mirror.
  local dest="$1" sub="$2" fname="$3"
  if [ -f "$dest" ]; then
    echo "$fname already exists, skipping."
    return 0
  fi
  echo "Downloading $fname ..."
  mkdir -p "$(dirname "$dest")"

  hf_fast "$dest" "$OFFICIAL_REPO" "$sub" && { echo "$fname done ($(du -h "$dest" | cut -f1))."; return 0; }
  echo "  official via hf failed (gated without HF_TOKEN?), trying mirror..."
  hf_fast "$dest" "$MIRROR_REPO" "$fname" && { echo "$fname done ($(du -h "$dest" | cut -f1))."; return 0; }

  echo "  fast downloader unavailable or failed, falling back to wget (resumable)..."
  fetch_hf "$dest" "$OFFICIAL_REPO" "$sub"
  if ! file_ok "$dest"; then
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

# If the workspace copy is missing, seed it from whatever ComfyUI the image ships.
# RunPod ComfyUI images do not all use the same layout, so probe rather than assume
# /opt/comfyui-baked exists: installing into a directory ComfyUI never reads fails
# silently and looks exactly like "the script never ran".
if [ ! -f "$COMFYUI_PATH/main.py" ]; then
  BAKED=""
  for cand in /opt/comfyui-baked /opt/ComfyUI /comfyui /ComfyUI /workspace/ComfyUI; do
    if [ -f "$cand/main.py" ]; then BAKED="$cand"; break; fi
  done
  if [ -z "$BAKED" ]; then
    echo "ComfyUI not in a known location, searching the filesystem..."
    FOUND="$(find / -maxdepth 5 -name main.py -path '*omfy*' \
      -not -path '*/custom_nodes/*' 2>/dev/null | head -1)"
    [ -n "$FOUND" ] && BAKED="$(dirname "$FOUND")"
  fi

  if [ -z "$BAKED" ] || [ ! -f "$BAKED/main.py" ]; then
    echo "FATAL: could not find a ComfyUI install (no main.py anywhere expected)."
    echo "       Check the container image, or set COMFYUI_PATH as a template env var."
    exit 1
  fi

  if [ "$BAKED" = "/opt/comfyui-baked" ]; then
    echo "First time setup: copying baked ComfyUI to $COMFYUI_PATH ..."
    rm -rf "$COMFYUI_PATH"
    mkdir -p "$(dirname "$COMFYUI_PATH")"
    cp -r "$BAKED" "$COMFYUI_PATH" || { echo "FATAL: copy failed."; exit 1; }
  else
    # Image keeps ComfyUI somewhere else: install into it directly rather than
    # copying, so models and workflows land where ComfyUI will actually read them.
    echo "Using the image's existing ComfyUI at $BAKED"
    COMFYUI_PATH="$BAKED"
  fi
fi

if [ ! -f "$COMFYUI_PATH/main.py" ]; then
  echo "FATAL: $COMFYUI_PATH/main.py missing after setup - aborting."
  exit 1
fi
echo "ComfyUI path: $COMFYUI_PATH"

# Use the interpreter ComfyUI actually runs from. The runpod/comfyui:cuda12.8 image
# keeps ComfyUI in its own venv, so a bare `pip install` puts packages in the system
# python where ComfyUI never sees them.
PY="$COMFYUI_PATH/.venv-cu128/bin/python"
[ -x "$PY" ] || PY="$COMFYUI_PATH/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
PIP="$PY -m pip"
echo "Using python: $PY"

# Fast multi-threaded downloader for the ~47GB of weights. Falls back to wget if the
# install or the CLI lookup fails, so this is an optimisation, never a hard dependency.
echo "Installing fast downloader (huggingface_hub + hf_transfer)..."
$PIP install -q -U "huggingface_hub[hf_transfer]" || \
  echo "WARN: could not install hf_transfer; downloads will use wget instead."
HF_BIN="$(dirname "$PY")/hf"
[ -x "$HF_BIN" ] || HF_BIN="$(dirname "$PY")/huggingface-cli"
[ -x "$HF_BIN" ] || HF_BIN="$(command -v hf || command -v huggingface-cli || true)"
if [ -n "$HF_BIN" ] && [ -x "$HF_BIN" ]; then
  echo "Fast downloader: $HF_BIN"
else
  HF_BIN=""
  echo "Fast downloader unavailable, using wget."
fi

# 0. Update ComfyUI core so LTX-2.5 native nodes + workflow templates exist
if [ "$UPDATE_COMFYUI" = "1" ]; then
  echo "Updating ComfyUI core (needed for native LTX-2.5 support)..."
  if [ -d "$COMFYUI_PATH/.git" ]; then
    # The baked checkout can sit on a detached HEAD/tag, where --ff-only has nothing to
    # fast-forward; fetch + hard-reset to the remote default branch handles both cases.
    if ! git -C "$COMFYUI_PATH" pull --ff-only; then
      echo "Fast-forward failed, resetting to origin's default branch..."
      DEFAULT_BRANCH="$(git -C "$COMFYUI_PATH" remote show origin 2>/dev/null \
        | sed -n 's/.*HEAD branch: //p')"
      DEFAULT_BRANCH="${DEFAULT_BRANCH:-master}"
      git -C "$COMFYUI_PATH" fetch --depth 1 origin "$DEFAULT_BRANCH" && \
        git -C "$COMFYUI_PATH" reset --hard "origin/$DEFAULT_BRANCH" || \
        echo "WARN: could not update ComfyUI, keeping baked version."
    fi
  else
    echo "WARN: $COMFYUI_PATH is not a git repo, cannot pull updates."
  fi

  # Pulling new core code WITHOUT its new dependencies is the main way this template
  # breaks: recent ComfyUI added packages (comfy-kitchen, comfy-aimdo, comfy-angle,
  # blake3, av>=17) and pins exact frontend/workflow-template versions. Installing the
  # repo's own requirements.txt keeps code and deps on the same version.
  if [ -f "$COMFYUI_PATH/requirements.txt" ]; then
    echo "Installing ComfyUI core requirements..."
    $PIP install -q -r "$COMFYUI_PATH/requirements.txt" || \
      echo "WARN: some core requirements failed to install; ComfyUI may not start."
  else
    $PIP install -q -U comfyui-frontend-package comfyui-workflow-templates || \
      echo "WARN: could not update UI/workflow-template packages."
  fi
fi

# 1. Move custom nodes to the official ComfyUI directory
echo "Installing custom nodes..."
mkdir -p "$COMFYUI_PATH/custom_nodes"
# Reference-flow compatibility: use pre-cloned repo nodes if the template provided them
if [ -d /tmp/temp_repo/custom_nodes ]; then
  cp -r /tmp/temp_repo/custom_nodes/* "$COMFYUI_PATH/custom_nodes/" 2>/dev/null || true
fi
# Official Lightricks node pack - optional, see INSTALL_LTXVIDEO_NODES above.
if [ "$INSTALL_LTXVIDEO_NODES" = "1" ]; then
  if [ ! -d "$COMFYUI_PATH/custom_nodes/ComfyUI-LTXVideo" ]; then
    git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo \
      "$COMFYUI_PATH/custom_nodes/ComfyUI-LTXVideo" || \
      echo "WARN: could not clone ComfyUI-LTXVideo; native LTX-2.5 nodes still work."
  fi
else
  echo "Skipping ComfyUI-LTXVideo (INSTALL_LTXVIDEO_NODES=0); bundled workflows use core nodes only."
fi

# 2. Automatically find and install requirements for your custom nodes
echo "Installing node requirements..."
find "$COMFYUI_PATH/custom_nodes/" -name "requirements.txt" -exec $PIP install -r {} \;

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
    hf_fast "$PE_DEST" "$ENHANCER_REPO" "text_encoders/gemma4_e2b_it_int8_convrot.safetensors" || \
    hf_fast "$PE_DEST" "$ENHANCER_REPO_FALLBACK" "gemma4_e2b_it_int8_convrot.safetensors" || true
    if ! file_ok "$PE_DEST"; then
      echo "Fast downloader failed, falling back to wget..."
      fetch_hf "$PE_DEST" "$ENHANCER_REPO" "text_encoders/gemma4_e2b_it_int8_convrot.safetensors"
    fi
    if ! file_ok "$PE_DEST"; then
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