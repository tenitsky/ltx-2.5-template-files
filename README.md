# LTX 2.5 — RunPod ComfyUI Template

One-shot template that installs and runs **Lightricks LTX-2.5** (22B video + synced-audio model, natively supported in ComfyUI) on RunPod. Based on the same flow as `reference-setup.sh` (Z-Image template).

On first pod boot it automatically:

1. Copies the baked ComfyUI from `/opt/comfyui-baked` to `/workspace/runpod-slim/ComfyUI` (self-healing, skip-if-present)
2. Updates ComfyUI core + workflow templates so the **native LTX-2.5 nodes** exist
3. Installs the official `ComfyUI-LTXVideo` custom node pack (Lightricks)
4. Downloads all LTX-2.5 model files into the correct `models/` folders (skip-if-exists, resumable)
5. Hands over to `/start.sh` (ComfyUI on port 3000)

## Models pulled

| File | Size | Destination (`ComfyUI/models/`) |
|---|---|---|
| `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | 21.5 GB | `diffusion_models/` |
| `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | 15.4 GB | `text_encoders/` |
| `ltx-2.5-video-vae-bf16.safetensors` | 1.5 GB | `vae/` |
| `ltx-2.5-audio-vae-bf16.safetensors` | 0.37 GB | `vae/` |
| `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | 0.26 GB | `latent_upscale_models/` |
| `ltx-2.5-duration-head-bf16.safetensors` | 4 MB | `model_patches/` |
| `ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors` *(optional)* | 0.26 GB | `latent_upscale_models/` |
| `gemma4_e2b_it_int8_convrot.safetensors` *(optional prompt enhancer)* | 8.1 GB | `text_encoders/` |

Core set ≈ **39 GB**; with both optional files ≈ **47 GB**. Use a **64 GB volume minimum, 80 GB recommended** (volume also stores the ComfyUI copy + your outputs).

## RunPod template settings

| Setting | Value |
|---|---|
| Base image | `runpod/comfyui:cuda12.8` |
| Container disk | 5 GB |
| Volume mount | `/workspace` |
| Volume size | 64–80 GB |
| Ports | HTTP `3000` (ComfyUI), HTTP `8888` (JupyterLab, boots automatically), HTTP `8080` (FileBrowser) |
| Env vars | `HF_TOKEN` (recommended), `JUPYTER_PASSWORD`, `FILEBROWSER_PASSWORD` (see below) |

**Docker command** (host `setup.sh` in a GitHub repo, e.g. the same repo you used for the Z-Image template):

```bash
bash -c "git clone --depth 1 https://github.com/tenitsky/ltx-2.5-template-files /tmp/temp_repo && bash /tmp/temp_repo/setup.sh"
```

## Hugging Face token (important)

The official weights repo [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5) is **license-gated**:

1. Sign in on Hugging Face, open the repo, click **Agree and Access** (auto-approval).
2. Create a **read** token: https://huggingface.co/settings/tokens
3. Add it to the RunPod template as env var `HF_TOKEN`.

Without `HF_TOKEN` the script still works — it automatically falls back to an ungated mirror of the same files (`lxxxy6/LTX-2.5`). Recommend using the token so downloads come from the official source.

## After it boots

- Open ComfyUI (port 3000) → **Workflow → Browse Templates** → search **"LTX-2.5"** for the three native workflows: **Text to Video (T2V)**, **Image to Video (I2V)**, **FLF2V** (first/last frame).
- Or use the **bundled lipsync workflow** (see next section).
- First boot downloads ~40–47 GB from HF; later boots skip everything and start in seconds (files persist on the network volume).
- Re-runs are safe: every step is skip-if-present and downloads resume (`wget -c`).

## Repository layout

```
├── setup.sh                      # RunPod boot script (Docker Command target)
├── README.md
└── workflows/                    # auto-installed into ComfyUI's Workflows menu on boot
    └── lipsync_i2v_workflow.json # LTX-2.5: person photo + dialogue script → talking video
```

## Bundled lipsync workflow (`workflows/lipsync_i2v_workflow.json`)

Person photo + dialogue script → talking video with **lip-synced speech audio**, built on the official ComfyUI **LTX-2.5 Image-to-Video** template. Uses **native nodes only** — no extra custom nodes required. The setup script copies everything in `workflows/` into `ComfyUI/user/default/workflows/`, so it appears in ComfyUI's **Workflows** sidebar on first boot.

**How to use:**

1. Open ComfyUI → **Workflows** sidebar (left panel) → `lipsync_i2v_workflow`. (Or `Workflow → Open` and select the JSON file manually.)
2. In the **Load First Frame** node, upload your person image (the template expects `person_lipsync.png`, but any uploaded name works — just select it).
3. The main **Image to Video (LTX-2.5)** node holds the prompt — replace the dialogue in quotes with your script:
   > ...The person says: "**YOUR SCRIPT LINE HERE, SPOKEN NATURALLY**"...
   - Keep the "Use the provided start image as the first frame" anchor phrase.
   - Describe delivery (tone, pacing, voice character) — the model generates the speech audio and lip movement from the text.
   - Keep scripts short: roughly one sentence per 2–3 seconds of video (default duration is 5s).
4. Knobs on the main node: `prompt_enhance` (leave **off** for scripted dialogue), `duration` (s), `width/height`, seed, frame rate; the **ResolutionSelector** node sets aspect/size.
5. **Queue** → result lands in `ComfyUI/output/` as `LTX-2.5_lipsync_*.mp4` **with audio**.

**Want to drive lips from your own voiceover file instead of text?** That's the audio-conditioned flow (`image + audio → video`); LTX-2.3's IA2V template/nodes cover it — native LTX-2.5 templates are T2V / I2V / FLF2V.

## GPU guidance

| GPU | Notes |
|---|---|
| 48 GB (RTX A6000 / RTX PRO 6000 / L40S) | Comfortable, recommended |
| 24 GB (RTX 4090 / 3090) | Works with the int8-convrot (distilled) files this template pulls; keep resolutions moderate |
| < 24 GB | Consider GGUF quants instead (see below) |

## Template env vars (all optional)

| Var | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | *(empty)* | Hugging Face read token for the gated official repo |
| `JUPYTER_PASSWORD` | *(empty)* | Token for JupyterLab on port 8888 (always boots; set a value!) |
| `FILEBROWSER_PASSWORD` | `adminadmin12` | Password for FileBrowser on port 8080 (user `admin`) — **change the default** |
| `UPDATE_COMFYUI` | `1` | `0` = skip ComfyUI core/UI update |
| `DOWNLOAD_PROMPT_ENHANCER` | `1` | `0` = skip 8.1 GB prompt-enhancer encoder |
| `DOWNLOAD_TEMPORAL_UPSCALER` | `1` | `0` = skip temporal upscaler |

## What boots in the container (from the image's `start.sh`)

The `runpod/comfyui` image (`runpod-workers/comfyui-base` source) starts, in order:

1. **FileBrowser** — port `8080`, root `/workspace` (web file manager; user `admin`)
2. **JupyterLab** — port `8888`, root `/workspace`, token = `JUPYTER_PASSWORD`
3. Custom ComfyUI args file: `/workspace/runpod-slim/comfyui_args.txt` (one flag per line, auto-applied)
4. **ComfyUI** — port `3000`; if ComfyUI crashes, the pod stays alive so SSH/Jupyter/FileBrowser remain reachable for debugging

Access everything via the pod's **Connect** menu in the RunPod console.

## References

- ComfyUI docs: https://docs.comfy.org/tutorials/video/ltx/ltx-2-5
- Official weights: https://huggingface.co/Lightricks/LTX-2.5
- Node pack: https://github.com/Lightricks/ComfyUI-LTXVideo
- Low-VRAM alternative (manual): GGUF quants `Abiray/LTX-2.5-Distilled-GGUF` + `city96/ComfyUI-GGUF` custom node
