# LTX 2.5 — RunPod ComfyUI Template

One-shot template that installs and runs **Lightricks LTX-2.5** (22B video + synced-audio model, natively supported in ComfyUI) on RunPod. Based on the same flow as `reference-setup.sh` (Z-Image template).

On first pod boot it automatically:

1. Copies the baked ComfyUI from `/opt/comfyui-baked` to `/workspace/runpod-slim/ComfyUI` (self-healing, skip-if-present)
2. Updates ComfyUI core **and installs that commit's `requirements.txt`**, so the **native LTX-2.5 nodes** exist and the new core dependencies come with them
3. Installs the two bundled lipsync workflows into ComfyUI's Workflows sidebar
4. Downloads all LTX-2.5 model files into the correct `models/` folders (skip-if-exists, resumable)
5. Hands over to `/start.sh` (ComfyUI on port **8188**)

> **No custom nodes required.** Every bundled workflow runs on core ComfyUI nodes only. The Lightricks `ComfyUI-LTXVideo` pack is available but **off by default** (`INSTALL_LTXVIDEO_NODES=1` to enable) — its dependencies (`openimageio`, `diffusers`, a `transformers` bump) often fail to build and can disturb the baked environment.

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
| Base image | `runpod/comfyui:1.4.0-rc.164-comfyuiv0.35.0-cuda12.8` **(not `cuda12.8`)** |
| Container disk | 5 GB |
| Volume mount | `/workspace` |
| Volume size | 64–80 GB |
| Ports | HTTP `8188` (ComfyUI), HTTP `8888` (JupyterLab). `8080` (FileBrowser) is optional — it starts regardless, expose it only if you want the web file manager |
| Env vars | `HF_TOKEN` (recommended), `JUPYTER_PASSWORD`, `FILEBROWSER_PASSWORD` (see below) |

> **Pick the image tag carefully.** `runpod/comfyui:cuda12.8` ships ComfyUI **v0.30.0**, which does not have `LTXVDualCFGGuider` — a node both bundled workflows need. They will fail to load on it.
>
> Use a tag with ComfyUI **v0.35.0** or newer, e.g. `runpod/comfyui:1.4.0-rc.164-comfyuiv0.35.0-cuda12.8`.
>
> `setup.sh` deliberately does **not** update ComfyUI core. On these images `pip install -r ComfyUI/requirements.txt` can reinstall `torch` (it is unpinned there) and break the baked CUDA build, leaving ComfyUI unable to start. Getting a new enough ComfyUI from the image is the safe route.

**Container start command.** RunPod's DNS is occasionally not ready when the container starts, so the command pins public resolvers, waits, and retries the clone rather than dying on the first failure:

```json
{
  "entrypoint": ["bash", "-c"],
  "cmd": ["echo 'nameserver 8.8.8.8' > /etc/resolv.conf && echo 'nameserver 1.1.1.1' >> /etc/resolv.conf && sleep 10 && rm -rf /tmp/temp_repo && for i in {1..10}; do git clone --depth 1 https://github.com/tenitsky/ltx-2.5-template-files.git /tmp/temp_repo && break || sleep 5; done && chmod +x /tmp/temp_repo/setup.sh && /tmp/temp_repo/setup.sh"]
}
```

Minimal equivalent, if your networking is reliable:

```bash
bash -c "git clone --depth 1 https://github.com/tenitsky/ltx-2.5-template-files /tmp/temp_repo && bash /tmp/temp_repo/setup.sh"
```

## Hugging Face token (important)

The official weights repo [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5) is **license-gated**:

1. Sign in on Hugging Face, open the repo, click **Agree and Access** (auto-approval).
2. Create a **read** token: https://huggingface.co/settings/tokens
3. Add it to the RunPod template as env var `HF_TOKEN`.

Without `HF_TOKEN` the script still works — it falls back to an ungated mirror of the same files (`lxxxy6/LTX-2.5`) — but **set the token anyway**. Anonymous Hub requests are rate-limited (~120/h vs ~1000/h) and get lower throughput, so an anonymous ~47 GB pull can be throttled part-way through. `huggingface_hub` says as much on every unauthenticated run: *"Please set a HF_TOKEN to enable higher rate limits and faster downloads."*

## After it boots

- Open ComfyUI (port 8188) → **Workflow → Browse Templates** → search **"LTX-2.5"** for the three native workflows: **Text to Video (T2V)**, **Image to Video (I2V)**, **FLF2V** (first/last frame).
- Or open the **Workflows** sidebar for the two bundled lipsync workflows — `lipsync_audio_ia2v_workflow` (photo + your audio file) and `lipsync_i2v_workflow` (photo + written script). See the sections below.
- First boot downloads ~40–47 GB from HF; later boots skip everything and start in seconds (files persist on the network volume).
- Downloads use **`hf_transfer`** (multi-threaded, typically several times faster than a single `wget` stream). Each file tries, in order: official repo → mirror via `hf`, then official → mirror via `wget`. The `wget` stage resumes partial files, so a dropped connection is never fatal.
- Re-runs are safe: every step is skip-if-present.

## Repository layout

```
├── setup.sh                             # RunPod boot script (Docker Command target)
├── README.md
└── workflows/                           # auto-installed into ComfyUI's Workflows menu on boot
    ├── lipsync_audio_ia2v_workflow.json # photo + YOUR audio file → lipsynced video
    └── lipsync_i2v_workflow.json        # photo + written dialogue → talking video (voice generated)
```

**Which one do you want?**

| You have | Use |
|---|---|
| A photo **and a voiceover/audio file** → lips must match *that* recording | `lipsync_audio_ia2v_workflow.json` |
| A photo and a **written script**, happy for the model to generate the voice | `lipsync_i2v_workflow.json` |

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

**Want to drive lips from your own voiceover file instead?** Use the workflow below.

## Audio-driven lipsync workflow (`workflows/lipsync_audio_ia2v_workflow.json`)

**Person photo + your own audio file → video whose lips match that recording.**

Comfy ships an official image+audio→video (IA2V) template for **LTX-2.3** but not for LTX-2.5 (the stock 2.5 templates are T2V / I2V / FLF2V, where audio is *generated* from the prompt). This workflow ports the 2.3 IA2V technique onto the LTX-2.5 two-stage I2V graph, so you get 2.5 quality with your own audio track.

**How it works:** LTX-2.5 is a joint audio+video model — it denoises one latent containing both streams. The stock I2V workflow seeds the audio half with `LTXVEmptyLatentAudio` (empty → model invents the voice). This workflow instead encodes your file with `LTXVAudioVAEEncode` and pins it using `SetLatentNoiseMask` with a `SolidMask` of value **0**. A zero noise mask means "never denoise this", so your audio survives untouched and the sampler can only generate the *video* that fits it — which is what produces the lipsync. The mask propagates through the second (upscale) stage too, so the audio stays locked end to end.

```
LoadAudio → TrimAudioDuration → LTXVAudioVAEEncode ─┐
                                 SolidMask(0) ──→ SetLatentNoiseMask
                                                    └→ LTXVConcatAVLatent → sampler → …
LoadImage → LTXVPreprocess → LTXVImgToVideoInplace ──┘
```

**How to use:**

1. ComfyUI → **Workflows** sidebar → `lipsync_audio_ia2v_workflow`.
2. **Load Image** — your person photo (front-facing, face clearly visible).
3. **Load Voiceover / Dialogue Audio** — upload your `.mp3` / `.wav` / `.flac`. Use **stereo**; mono clips frequently produce weak or no lipsync.
4. **`duration`** (on the main node) — seconds of video. The audio is auto-trimmed to this same value, so **set it to your clip's length** (or shorter). Audio shorter than `duration` leaves the tail unsynced.
5. **`prompt`** — describe the person, framing and delivery. **Do not write the dialogue** — the words come from your audio file, not the prompt.
6. Leave **`prompt_enhance` off**; it rewrites your prompt and works against the lipsync.
7. **Queue** → `ComfyUI/output/LTX-2.5_lipsync_*.mp4`, with your audio muxed in.

**If the second stage OOMs**, add `--reserve-vram 1` to `/workspace/runpod-slim/comfyui_args.txt` (one flag per line, picked up automatically) and restart the pod.

**Uses core ComfyUI nodes only** — `LoadAudio`, `TrimAudioDuration`, `LTXVAudioVAEEncode`, `SolidMask`, `SetLatentNoiseMask`, `LTXVConcatAVLatent` and friends all ship with ComfyUI. Nothing extra to install.

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
| `UPDATE_COMFYUI` | `1` | `0` = skip ComfyUI core update + core `requirements.txt` install |
| `DOWNLOAD_PROMPT_ENHANCER` | `1` | `0` = skip 8.1 GB prompt-enhancer encoder. Leave at `1`: both bundled workflows contain a `CLIPLoader` pointing at that file, and ComfyUI validates it even when the enhancer is switched off |
| `DOWNLOAD_TEMPORAL_UPSCALER` | `1` | `0` = skip temporal upscaler |
| `INSTALL_LTXVIDEO_NODES` | `0` | `1` = also clone the Lightricks `ComfyUI-LTXVideo` pack. Not needed by anything bundled here |
| `COMFYUI_PATH` | `/workspace/runpod-slim/ComfyUI` | Override only if your image uses a different layout. The script auto-detects ComfyUI and aborts loudly if it can't find it |
| `HF_HOME` | `/workspace/.cache/huggingface` | HF cache location — kept on the volume, never the 5 GB container disk |

## What boots in the container (from the image's `start.sh`)

The `runpod/comfyui` image (`runpod-workers/comfyui-base` source) starts, in order:

1. **FileBrowser** — port `8080`, root `/workspace` (web file manager; user `admin`)
2. **JupyterLab** — port `8888`, root `/workspace`, token = `JUPYTER_PASSWORD`
3. Custom ComfyUI args file: `/workspace/runpod-slim/comfyui_args.txt` (one flag per line, auto-applied)
4. **ComfyUI** — port `8188` (launched with `--listen 0.0.0.0 --port 8188 --enable-cors-header`); if ComfyUI crashes, the pod stays alive so SSH/Jupyter/FileBrowser remain reachable for debugging

ComfyUI runs from its own virtualenv (`$COMFYUI_PATH/.venv-cu128`), not the system Python. `setup.sh` detects it and routes every `pip install` through `$PY -m pip`, so packages land where ComfyUI can import them. If you install anything by hand over SSH, use the same interpreter:

```bash
/workspace/runpod-slim/ComfyUI/.venv-cu128/bin/python -m pip install <package>
```

Access everything via the pod's **Connect** menu in the RunPod console.

## References

- ComfyUI docs: https://docs.comfy.org/tutorials/video/ltx/ltx-2-5
- Official LTX-2.3 image+audio→video template (the technique ported here): https://comfy.org/workflows/video_ltx2_3_ia2v-adca306765ce/
- Official weights: https://huggingface.co/Lightricks/LTX-2.5
- Node pack: https://github.com/Lightricks/ComfyUI-LTXVideo
- Low-VRAM alternative (manual): GGUF quants `Abiray/LTX-2.5-Distilled-GGUF` + `city96/ComfyUI-GGUF` custom node
