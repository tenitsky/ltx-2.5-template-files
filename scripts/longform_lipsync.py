#!/usr/bin/env python3
"""
Long-form LTX-2.5 lipsync: turn one portrait + a long audio track into a video of
arbitrary length, by chaining fixed-length LTX-2.5 generations.

LTX-2.5 tops out around 10s per generation, so a 7-minute track becomes ~42 chunks.
This script drives ComfyUI's HTTP API to produce them in sequence and stitches the
result, handling the three things that silently ruin a naive chain:

  1. Frame accounting. LTX renders duration*fps + 1 frames. Concatenating 42 chunks
     with an extra frame each accumulates ~1.75s of audio drift. Every chunk except
     the last is trimmed back to exactly duration*fps frames.

  2. Audio quality and sync. Each chunk's audio has been through the audio VAE and
     comes back slightly degraded. Since we already have the original track, the
     final mux uses that and discards the generated audio entirely. Sync is then
     exact by construction rather than by luck.

  3. Identity drift. Chaining last-frame -> next-first-frame keeps motion continuous
     but lets the face wander over dozens of hops. --reanchor-every resets the start
     frame back to the original portrait periodically, trading a small visual jump
     for a face that still matches at minute seven.

Usage:
  python longform_lipsync.py \
      --api-workflow lipsync_api.json \
      --image portrait.png \
      --audio voiceover.mp3 \
      --out final.mp4

Get lipsync_api.json from ComfyUI: open the bundled lipsync_audio_ia2v_workflow,
then Workflow -> Export (API). The UI-format JSON in workflows/ will NOT work here;
the API format is what /prompt accepts.

Requires: ffmpeg + ffprobe on PATH, `pip install requests`, and a running ComfyUI.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import uuid

import requests

FPS = 24  # at 24fps every whole-second duration yields a legal frame count


# --------------------------------------------------------------------------- utils
def run(cmd, **kw):
    """Run a command, raising with the tail of stderr so failures are readable."""
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{p.stderr[-2000:]}")
    return p.stdout


def audio_duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", path])
    return float(out.strip())


def find_silences(path, noise_db=-32, min_silence=0.30):
    """Return midpoints of detected silences, used as preferred cut points.

    Cutting mid-word is the most audible artifact in a chained render, so we bias
    chunk boundaries toward natural pauses instead of a fixed grid.
    """
    p = subprocess.run(
        ["ffmpeg", "-i", path, "-af",
         f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"],
        capture_output=True, text=True)
    starts = [float(m) for m in re.findall(r"silence_start: ([0-9.]+)", p.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end: ([0-9.]+)", p.stderr)]
    return [(s + e) / 2 for s, e in zip(starts, ends)]


def plan_chunks(total, target, min_len, max_len, silences):
    """Split [0,total] into whole-second spans, preferring cuts at silences.

    Whole seconds matter: the workflow computes frames as duration*fps + 1, and a
    non-integer duration would round down inside ComfyUI and shorten the clip.
    """
    spans, pos = [], 0.0
    while total - pos > max_len:
        ideal = pos + target
        lo, hi = pos + min_len, pos + max_len
        cands = [s for s in silences if lo <= s <= hi]
        cut = min(cands, key=lambda s: abs(s - ideal)) if cands else ideal
        cut = pos + max(min_len, min(max_len, round(cut - pos)))  # snap to whole second
        if cut <= pos:
            cut = pos + target
        spans.append((pos, int(round(cut - pos))))
        pos = cut
    remaining = int(round(total - pos))
    if remaining >= 1:
        spans.append((pos, remaining))

    # A stubby tail (say 2s) generates poorly and looks like an offcut. Fold it into
    # the previous chunk when that fits, otherwise split the last two evenly so both
    # land in range.
    if len(spans) >= 2 and spans[-1][1] < min_len:
        (p_start, p_dur), (_, l_dur) = spans[-2], spans[-1]
        combined = p_dur + l_dur
        if combined <= max_len:
            spans[-2:] = [(p_start, combined)]
        else:
            first = combined // 2
            second = combined - first
            if first >= min_len and second >= min_len:
                spans[-2:] = [(p_start, first), (p_start + first, second)]
    return spans


# ----------------------------------------------------------------- workflow patching
def find_nodes(wf, class_type=None, title=None):
    hits = []
    for nid, node in wf.items():
        if class_type and node.get("class_type") != class_type:
            continue
        if title and node.get("_meta", {}).get("title") != title:
            continue
        hits.append(nid)
    return hits


def patch(wf, image_name, audio_name, duration, seed, prefix):
    """Set the per-chunk inputs on an exported API workflow.

    Nodes are located by class_type (plus title where several share a class), which
    survives re-exports better than hardcoded numeric ids.
    """
    wf = json.loads(json.dumps(wf))  # deep copy: each chunk gets its own graph

    for nid in find_nodes(wf, "LoadImage"):
        wf[nid]["inputs"]["image"] = image_name
    for nid in find_nodes(wf, "LoadAudio"):
        wf[nid]["inputs"]["audio"] = audio_name

    dur = find_nodes(wf, "PrimitiveInt", "Duration") or [
        n for n in find_nodes(wf, "PrimitiveInt")
        if wf[n]["inputs"].get("value") in (5, 6, 8, 10)]
    if not dur:
        raise SystemExit("Could not find the Duration node. Re-export the API JSON, "
                         "or check that the duration primitive is titled 'Duration'.")
    for nid in dur[:1]:
        wf[nid]["inputs"]["value"] = duration

    # Both sampling stages have their own noise node; seed them together so a chunk
    # is reproducible from its seed alone.
    for nid in find_nodes(wf, "RandomNoise"):
        wf[nid]["inputs"]["noise_seed"] = seed
    for nid in find_nodes(wf, "SaveVideo"):
        wf[nid]["inputs"]["filename_prefix"] = prefix
    return wf


# ------------------------------------------------------------------------ comfy api
class Comfy:
    def __init__(self, host):
        self.host = host.rstrip("/")
        self.cid = str(uuid.uuid4())

    def submit(self, wf):
        r = requests.post(f"{self.host}/prompt",
                          json={"prompt": wf, "client_id": self.cid}, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"ComfyUI rejected the workflow ({r.status_code}):\n"
                               f"{r.text[:1500]}")
        return r.json()["prompt_id"]

    def wait(self, pid, timeout=3600, poll=3):
        """Block until the prompt leaves the queue, then return its history entry."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            h = requests.get(f"{self.host}/history/{pid}", timeout=30).json()
            if pid in h:
                entry = h[pid]
                status = entry.get("status", {})
                if status.get("status_str") == "error" or status.get("completed") is False:
                    raise RuntimeError(f"Generation failed:\n"
                                       f"{json.dumps(status, indent=2)[:2000]}")
                return entry
            time.sleep(poll)
        raise TimeoutError(f"Chunk did not finish within {timeout}s")

    def fetch(self, item, dest):
        q = urllib.parse.urlencode({"filename": item["filename"],
                                    "subfolder": item.get("subfolder", ""),
                                    "type": item.get("type", "output")})
        r = requests.get(f"{self.host}/view?{q}", timeout=300)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)


def result_video(entry):
    """Pull the produced video out of a history entry.

    ComfyUI reports video outputs under different keys depending on the save node,
    so check the known ones rather than assuming a single shape.
    """
    for node_out in entry.get("outputs", {}).values():
        for key in ("videos", "gifs", "images", "audio"):
            for item in node_out.get(key, []) or []:
                if str(item.get("filename", "")).lower().endswith(
                        (".mp4", ".webm", ".mkv")):
                    return item
    return None


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-workflow", required=True,
                    help="workflow exported via Workflow -> Export (API)")
    ap.add_argument("--image", required=True, help="portrait, front-facing")
    ap.add_argument("--audio", required=True, help="full-length voiceover (stereo)")
    ap.add_argument("--out", default="final.mp4")
    ap.add_argument("--comfy", default="http://127.0.0.1:8188")
    ap.add_argument("--comfy-input", default="/workspace/runpod-slim/ComfyUI/input",
                    help="ComfyUI's input dir; chunk audio and start frames go here")
    ap.add_argument("--work", default="./longform_work")
    ap.add_argument("--chunk", type=int, default=8, help="target chunk seconds")
    ap.add_argument("--min-chunk", type=int, default=5)
    ap.add_argument("--max-chunk", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--reanchor-every", type=int, default=0,
                    help="reset to the original portrait every N chunks (0 = never). "
                         "Fights identity drift at the cost of a small visual jump.")
    ap.add_argument("--resume", action="store_true",
                    help="skip chunks whose output already exists")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH")

    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    os.makedirs(args.comfy_input, exist_ok=True)
    comfy = Comfy(args.comfy)
    wf_template = json.load(open(args.api_workflow, encoding="utf-8"))

    total = audio_duration(args.audio)
    print(f"Audio: {total:.1f}s ({total/60:.1f} min)")
    print("Detecting pauses to cut on...")
    spans = plan_chunks(total, args.chunk, args.min_chunk, args.max_chunk,
                        find_silences(args.audio))
    print(f"Planned {len(spans)} chunks: "
          f"{', '.join(str(d) for _, d in spans[:12])}"
          f"{' ...' if len(spans) > 12 else ''}")

    est = len(spans) * 2.5
    print(f"Rough estimate: {est:.0f} min of GPU time (~2.5 min/chunk on 48GB).\n")

    # The portrait is the anchor; it stays available for re-anchoring.
    anchor = f"lf_anchor_{uuid.uuid4().hex[:6]}.png"
    shutil.copy(args.image, os.path.join(args.comfy_input, anchor))
    start_image = anchor

    produced = []
    for i, (start, dur) in enumerate(spans):
        out_mp4 = os.path.join(work, f"chunk_{i:04d}.mp4")
        trimmed = os.path.join(work, f"trim_{i:04d}.mp4")
        last_png = os.path.join(work, f"last_{i:04d}.png")

        if args.resume and os.path.exists(trimmed):
            print(f"[{i+1}/{len(spans)}] exists, skipping")
            produced.append(trimmed)
            if os.path.exists(last_png):
                nm = f"lf_start_{i:04d}.png"
                shutil.copy(last_png, os.path.join(args.comfy_input, nm))
                start_image = nm
            continue

        if args.reanchor_every and i and i % args.reanchor_every == 0:
            print(f"[{i+1}/{len(spans)}] re-anchoring to the original portrait")
            start_image = anchor

        # Slice this chunk's audio. wav keeps the model's encoder off a lossy decode.
        chunk_name = f"lf_audio_{i:04d}.wav"
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", str(dur),
             "-i", args.audio, "-ac", "2", "-ar", "44100",
             os.path.join(args.comfy_input, chunk_name)])

        print(f"[{i+1}/{len(spans)}] {start:.1f}s +{dur}s  start={start_image}")
        wf = patch(wf_template, start_image, chunk_name, dur,
                   args.seed + i, f"longform/chunk_{i:04d}")
        entry = comfy.wait(comfy.submit(wf))

        item = result_video(entry)
        if not item:
            raise SystemExit(
                f"Chunk {i} produced no video. Outputs seen: "
                f"{json.dumps(entry.get('outputs', {}))[:800]}")
        comfy.fetch(item, out_mp4)

        # Drop the trailing frame so the chunk is exactly dur*FPS frames. Without
        # this each chunk runs one frame long and the audio slides progressively
        # later across the full render.
        run(["ffmpeg", "-y", "-v", "error", "-i", out_mp4,
             "-frames:v", str(dur * FPS), "-an",
             "-c:v", "libx264", "-crf", "16", "-preset", "medium", trimmed])
        produced.append(trimmed)

        # Last frame becomes the next chunk's anchor, keeping motion continuous.
        run(["ffmpeg", "-y", "-v", "error", "-sseof", "-0.2", "-i", trimmed,
             "-update", "1", "-q:v", "2", last_png])
        nm = f"lf_start_{i:04d}.png"
        shutil.copy(last_png, os.path.join(args.comfy_input, nm))
        start_image = nm

    # Concatenate video only, then mux the untouched original audio. The generated
    # audio is a VAE reconstruction of the input, so the original is strictly better
    # and removes any chance of cumulative sync error.
    print("\nStitching...")
    listfile = os.path.join(work, "concat.txt")
    with open(listfile, "w", encoding="utf-8") as f:
        for p in produced:
            f.write(f"file '{os.path.abspath(p)}'\n")
    silent = os.path.join(work, "video_only.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", listfile, "-c", "copy", silent])
    run(["ffmpeg", "-y", "-v", "error", "-i", silent, "-i", args.audio,
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
         "-c:a", "aac", "-b:a", "192k", "-shortest", args.out])

    print(f"\nDone: {args.out} ({audio_duration(args.out):.1f}s)")
    print(f"Intermediates in {work} (delete when happy; --resume reuses them)")


if __name__ == "__main__":
    main()
