"""
LTX Longform - turn one portrait + a long audio track into a minutes-long lipsync
video, using ComfyUI's own batch queue instead of an external driver script.

ComfyUI's graph is a DAG: every node runs once per queue item, so no single node can
orchestrate 50 sequential generations without reimplementing the sampler. These three
nodes take the other route - each queue item renders one chunk, and ComfyUI's batch
count provides the loop:

    Split  -> feeds this chunk's audio + duration into the workflow
    Start  -> supplies the start frame (previous chunk's last frame, or the portrait)
    Write  -> saves the chunk, and on the final one stitches everything together

Set the chunk index to increment, queue with batch count = total_chunks, done.

Three things are handled at assembly time because they cannot be fixed afterwards:

  * LTX renders duration*fps + 1 frames. Keeping that extra frame on every chunk
    accumulates ~2s of progressive audio slip across a 7-minute render, so each chunk
    is written with exactly duration*fps frames.
  * Each chunk's generated audio is a VAE reconstruction of the input. The original
    track is strictly better and makes sync exact by construction, so the final mux
    uses it and the generated audio is discarded.
  * Cutting mid-word is the most audible artifact in a chained render, so chunk
    boundaries are biased toward pauses found in the waveform.

No dependencies beyond what ComfyUI already has, plus ffmpeg.
"""

import os
import shutil
import subprocess

import numpy as np
import torch

import folder_paths

try:
    # Returning this from a node blocks everything downstream, so an
    # out-of-range chunk finishes in milliseconds instead of sampling.
    from comfy_execution.graph_utils import ExecutionBlocker
except ImportError:  # pragma: no cover - older ComfyUI
    try:
        from comfy_execution.graph import ExecutionBlocker
    except ImportError:
        ExecutionBlocker = None

FPS_DEFAULT = 24


# --------------------------------------------------------------------------- helpers
def _session_dir(session):
    d = os.path.join(folder_paths.get_output_directory(), "ltx_longform", session)
    os.makedirs(d, exist_ok=True)
    return d


def _mono(waveform):
    """[B,C,T] -> 1-D numpy, averaged across batch and channels."""
    w = waveform
    if w.dim() == 3:
        w = w[0]
    if w.dim() == 2:
        w = w.mean(dim=0)
    return w.detach().cpu().float().numpy()


def _find_pauses(samples, sr, thresh_db=-34.0, min_len=0.25, win=0.02):
    """Midpoints of quiet stretches, computed from the waveform.

    Done in-memory rather than by shelling out to ffmpeg's silencedetect so the plan
    is deterministic and identical on every queue item - each chunk recomputes the
    plan independently, so it has to agree with the others exactly.
    """
    n = max(1, int(sr * win))
    usable = (len(samples) // n) * n
    if usable < n:
        return []
    frames = samples[:usable].reshape(-1, n)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    peak = float(rms.max())
    if peak <= 0:
        return []
    quiet = 20 * np.log10(rms / peak + 1e-12) < thresh_db

    pauses, run = [], 0
    for i, q in enumerate(quiet):
        if q:
            run += 1
            continue
        if run * win >= min_len:
            pauses.append(((i - run) + i) / 2 * win)
        run = 0
    if run * win >= min_len:
        pauses.append(((len(quiet) - run) + len(quiet)) / 2 * win)
    return pauses


def plan_chunks(total, target, min_len, max_len, pauses, pause_driven=False):
    """Split [0,total] into whole-second spans.

    Whole seconds matter: the workflow computes frames as duration*fps + 1, and
    ComfyUI floor-divides that back into latent frames, so a fractional duration
    silently yields a shorter clip than asked for.

    Two ways to choose a boundary:

    * target-driven (default) - aim for `target`, and accept a nearby pause if one
      falls inside [min_len, max_len].
    * pause_driven - ignore `target` and take the *furthest* pause still within
      max_len. Short speech segments get merged rather than generated separately,
      chunk lengths follow the speech rather than the clock, and every boundary
      lands in a pause. A time cut only happens when no pause is reachable at all,
      which is forced by the model's length ceiling rather than a preference.
    """
    spans, pos = [], 0.0
    while total - pos > max_len:
        lo, hi = pos + min_len, pos + max_len
        cands = [p for p in pauses if lo <= p <= hi]
        if pause_driven:
            # Furthest reachable pause: fewest boundaries, so fewer joins and -
            # in the FLF2V graph - fewer returns to the portrait pose.
            cut = max(cands) if cands else pos + max_len
        else:
            ideal = pos + target
            cut = min(cands, key=lambda p: abs(p - ideal)) if cands else ideal
        cut = pos + max(min_len, min(max_len, round(cut - pos)))
        if cut <= pos:
            cut = pos + (max_len if pause_driven else target)
        spans.append((pos, int(round(cut - pos))))
        pos = cut

    remaining = int(round(total - pos))
    if remaining >= 1:
        spans.append((pos, remaining))

    # A stubby tail generates poorly and reads as an offcut: fold it into the
    # previous chunk, or rebalance the last two so both land in range.
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


def _audio_basename(prompt, namer_id=None, default="run1"):
    """The LoadAudio selection, without directory or extension.

    Read from the prompt rather than taken as a link, so the file is chosen once
    in LoadAudio and the name follows automatically.
    """
    if not isinstance(prompt, dict):
        return default
    want = ""
    if namer_id is not None:
        n = prompt.get(str(namer_id)) or {}
        want = ((n.get("inputs") or {}).get("source_title") or "").strip()
    for node in prompt.values():
        if not isinstance(node, dict) or node.get("class_type") != "LoadAudio":
            continue
        if want and ((node.get("_meta") or {}).get("title") or "") != want:
            continue
        v = (node.get("inputs") or {}).get("audio")
        if isinstance(v, str) and v.strip():
            base = os.path.basename(v.strip().replace("\\", "/"))
            stem = os.path.splitext(base)[0].strip()
            if stem:
                return stem
    return default


def _session_from_prompt(prompt, default="run1"):
    """Read the session name off the Write node.

    Start Frame runs before Write, so it cannot take the value as a link without
    creating a cycle. Reading it out of the prompt keeps the name on exactly one
    node instead of requiring two widgets to be kept in sync by hand.
    """
    if isinstance(prompt, dict):
        for node in prompt.values():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") != "LTXLongformWrite":
                continue
            v = (node.get("inputs") or {}).get("session")
            if isinstance(v, str) and v.strip():
                return v.strip()
            # A widget converted to an input arrives as [node_id, slot]. Follow it
            # when it comes from the namer, otherwise Start Frame would fall back to
            # the default while Write used the real name - the chain would break
            # silently and every chunk would restart from the portrait.
            if isinstance(v, list) and len(v) == 2:
                src = prompt.get(str(v[0])) or {}
                if src.get("class_type") == "LTXLongformAudioName":
                    return _audio_basename(prompt, v[0], default)
    return default


def _save_anchor(frame, path_noext):
    """Store a chunk's last frame for the next chunk to start from.

    PNG rather than .npy: at 1280x720 a float32 array is 11MB, so an hour-long
    render would leave ~5.7GB of anchors on the volume for ~0.57GB of actual
    information. The frame is quantised to 8 bits when encoded to video anyway, so
    float32 precision buys nothing here - and a PNG can be opened, which makes
    drift visible instead of a matter of opinion.
    """
    arr = (frame.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(arr).save(path_noext + ".png")
        return path_noext + ".png"
    except Exception:
        np.save(path_noext + ".npy", frame.detach().cpu().numpy())
        return path_noext + ".npy"


def _anchor_base(session, chunk_index):
    """Anchors live in a hidden subfolder so the session folder shows only mp4s."""
    d = os.path.join(_session_dir(session), ".anchors")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"last_{chunk_index:04d}")


def _anchors_needed(prompt):
    """Whether any node will ever read a chunk's last frame.

    Only a Start Frame that chains needs them. At reanchor_every=1 every chunk
    starts from the portrait, and the FLF2V graph has no Start Frame at all - in
    both cases writing anchors is pure waste. When the value cannot be read (a
    linked widget, or no prompt), keep them: a missing anchor silently breaks the
    chain, an unneeded one costs a megabyte.
    """
    if not isinstance(prompt, dict):
        return True
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") != "LTXLongformStartFrame":
            continue
        r = (node.get("inputs") or {}).get("reanchor_every", 1)
        if not isinstance(r, int) or r != 1:
            return True
    return False


def _find_anchor(session, chunk_index):
    """Path (without extension) of a saved anchor, new location first."""
    new = os.path.join(_session_dir(session), ".anchors", f"last_{chunk_index:04d}")
    old = os.path.join(_session_dir(session), f"last_{chunk_index:04d}")
    for base in (new, old):          # old: sessions started before the move
        for ext in (".png", ".npy"):
            if os.path.exists(base + ext):
                return base
    return None


def _load_anchor(path_noext):
    """Read an anchor, accepting .npy from sessions started before the switch."""
    png = path_noext + ".png"
    if os.path.exists(png):
        from PIL import Image
        arr = np.array(Image.open(png).convert("RGB"), dtype=np.float32) / 255.0
        return arr
    npy = path_noext + ".npy"
    if os.path.exists(npy):
        return np.load(npy)
    return None


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError(
            "ffmpeg not found. On the RunPod template it is installed by setup.sh; "
            "otherwise: apt-get install -y ffmpeg")
    return exe


# ----------------------------------------------------------------------------- nodes
class LTXLongformSplit:
    """Slice one chunk out of a long track, and report how many chunks there are."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "chunk_index": ("INT", {"default": 0, "min": 0, "max": 100000,
                                        "tooltip": "Set this to increment, then queue "
                                                   "with batch count = total_chunks."}),
                "target_seconds": ("INT", {"default": 8, "min": 2, "max": 20}),
                "min_seconds": ("INT", {"default": 5, "min": 1, "max": 20}),
                "max_seconds": ("INT", {"default": 10, "min": 2, "max": 20}),
            },
            # Optional, not required: a new required input invalidates every
            # workflow saved before it existed.
            "optional": {
                "skip_existing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Skip chunks whose .mp4 is already on disk, costing "
                               "milliseconds each. Lets you resume an interrupted "
                               "render by queueing from 0 again - no need to work "
                               "out where it stopped. The final chunk always "
                               "re-renders so the stitch has something to fire on."}),
                "cut_mode": (["silence", "pause", "fixed"], {
                    "default": "silence",
                    "tooltip": "silence: aim for target_seconds, but snap to a pause "
                               "if one is in range. pause: ignore target_seconds and "
                               "let the speech decide - take the furthest pause "
                               "within max_seconds, merging segments too short to "
                               "generate. fixed: a strict grid, keeping pauses "
                               "mid-chunk; try it if speech starts early after a "
                               "pause."}),
            },
            # Needed to resolve the session name for the skip-existing check.
            "hidden": {"prompt": "PROMPT"},
        }

    RETURN_TYPES = ("AUDIO", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("audio_chunk", "duration", "total_chunks", "is_last")
    FUNCTION = "split"
    CATEGORY = "LTX Longform"

    def split(self, audio, chunk_index, target_seconds, min_seconds, max_seconds,
              cut_mode="silence", skip_existing=True, prompt=None):
        if min_seconds > max_seconds:
            min_seconds, max_seconds = max_seconds, min_seconds
        target_seconds = max(min_seconds, min(max_seconds, target_seconds))

        wav, sr = audio["waveform"], int(audio["sample_rate"])
        total = wav.shape[-1] / sr
        pauses = _find_pauses(_mono(wav), sr) if cut_mode != "fixed" else []
        if chunk_index == 0:
            # Worth surfacing: pause detection is thresholded against the track's
            # peak, so music or room tone under the voice can leave it finding
            # nothing, and cuts then fall back to a fixed grid anyway.
            print(f"[LTX Longform] cut_mode={cut_mode}, pauses found: {len(pauses)}")
        spans = plan_chunks(total, target_seconds, min_seconds, max_seconds, pauses,
                            pause_driven=(cut_mode == "pause"))
        if not spans:
            raise RuntimeError("Audio too short to split into chunks.")

        if chunk_index >= len(spans):
            # Past the end. Blocking here is what makes it safe to queue a
            # generous batch count without knowing the total in advance: the
            # surplus items cost milliseconds each instead of a full render.
            if ExecutionBlocker is not None:
                if chunk_index == len(spans):
                    print(f"[LTX Longform] all {len(spans)} chunks done - "
                          f"skipping surplus queue items.")
                return tuple(ExecutionBlocker(None) for _ in range(4))
            # No blocker available: fall back to repeating the last chunk.
            print("[LTX Longform] past the last chunk; stop the queue manually.")

        # Resume: a chunk already on disk does not need generating again. The last
        # chunk is exempt because it is what triggers the stitch.
        if (skip_existing and ExecutionBlocker is not None
                and chunk_index < len(spans) - 1):
            done = os.path.join(_session_dir(_session_from_prompt(prompt)),
                                f"chunk_{chunk_index:04d}.mp4")
            if os.path.exists(done):
                print(f"[LTX Longform] chunk {chunk_index + 1}/{len(spans)} "
                      f"already rendered - skipping.")
                return tuple(ExecutionBlocker(None) for _ in range(4))

        idx = min(chunk_index, len(spans) - 1)
        start, dur = spans[idx]
        a = int(round(start * sr))
        b = min(wav.shape[-1], a + dur * sr)
        seg = wav[..., a:b]

        print(f"[LTX Longform] chunk {idx + 1}/{len(spans)}  "
              f"{start:.1f}s +{dur}s  (track {total:.1f}s)")
        return ({"waveform": seg, "sample_rate": sr},
                dur, len(spans), idx >= len(spans) - 1)


class LTXLongformStartFrame:
    """Start frame for this chunk: the previous chunk's last frame, or the portrait.

    Chaining keeps motion continuous across chunks, but the face drifts a little at
    every hop. reanchor_every periodically returns to the original portrait, trading a
    small visual jump for an identity that still matches at minute seven.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "portrait": ("IMAGE",),
                "chunk_index": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "reanchor_every": ("INT", {"default": 1, "min": 0, "max": 1000,
                                           "tooltip": "Start every Nth chunk from the "
                                                      "portrait instead of the previous "
                                                      "chunk's last frame. 1 = every "
                                                      "chunk, so no drift at all. Raise "
                                                      "it for smoother motion across "
                                                      "boundaries at the cost of drift. "
                                                      "0 = never re-anchor."}),
            },
            # Session name is taken from the Write node so it is set in one place.
            "hidden": {"prompt": "PROMPT"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("start_frame",)
    FUNCTION = "pick"
    CATEGORY = "LTX Longform"

    @classmethod
    def IS_CHANGED(cls, chunk_index, reanchor_every, prompt=None, **kw):
        # The file on disk changes between queue items while the widget values do
        # not, so the previous chunk's frame has to be part of the cache key or
        # ComfyUI would serve chunk N-1's result again.
        session = _session_from_prompt(prompt)
        base = _find_anchor(session, chunk_index - 1)
        p = next((base + e for e in (".png", ".npy")
                  if base and os.path.exists(base + e)), None)
        return f"{session}:{chunk_index}:{os.path.getmtime(p) if p else 0}"

    def pick(self, portrait, chunk_index, reanchor_every, prompt=None):
        session = _session_from_prompt(prompt)
        if chunk_index == 0:
            print("[LTX Longform] chunk 0: starting from the portrait")
            return (portrait,)
        if reanchor_every and chunk_index % reanchor_every == 0:
            print(f"[LTX Longform] chunk {chunk_index}: re-anchoring to the portrait")
            return (portrait,)

        prev = _find_anchor(session, chunk_index - 1)
        arr = _load_anchor(prev) if prev else None
        if arr is None:
            print(f"[LTX Longform] WARNING: no last frame from chunk "
                  f"{chunk_index - 1} in session '{session}'; falling back to the "
                  f"portrait. (Queue chunks in order.)")
            return (portrait,)
        return (torch.from_numpy(arr).unsqueeze(0),)


class LTXLongformWrite:
    """Write this chunk, and on the final one stitch the whole render together.

    Takes decoded frames rather than a VIDEO so the trailing frame can be dropped
    before anything is encoded.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "original_audio": ("AUDIO", {"tooltip": "The FULL track, not the "
                                                        "chunk - used for the final mux."}),
                "chunk_index": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "total_chunks": ("INT", {"default": 1, "min": 1, "max": 100000}),
                "duration": ("INT", {"default": 8, "min": 1, "max": 60}),
                "fps": ("INT", {"default": FPS_DEFAULT, "min": 1, "max": 120}),
                "session": ("STRING", {"default": "run1",
                                       "tooltip": "Folder for this render under "
                                                  "output/ltx_longform/. Set it here "
                                                  "only - Start Frame picks it up "
                                                  "automatically. Change it to start "
                                                  "a fresh render."}),
                "filename": ("STRING", {"default": "longform_final.mp4"}),
            },
            "optional": {
                # Muxed into the per-chunk file only. The finished render always
                # uses the full original track, so this cannot affect it - it is
                # here so a single chunk can be checked for sync on its own.
                "chunk_audio": ("AUDIO", {"tooltip": "Connect Split's audio_chunk "
                                                     "to make chunks playable."}),
                "stop_when_done": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "After the final chunk, clear the pending queue so a "
                               "generous batch count (or Run Instant) stops by "
                               "itself. This clears ALL pending items, including "
                               "unrelated jobs."}),
            },
            # Used to decide whether the next chunk will need this one's last frame.
            "hidden": {"prompt": "PROMPT"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "write"
    OUTPUT_NODE = True
    CATEGORY = "LTX Longform"

    def write(self, images, original_audio, chunk_index, total_chunks,
              duration, fps, session, filename, chunk_audio=None,
              stop_when_done=True, prompt=None):
        ff = _ffmpeg()
        sdir = _session_dir(session)

        # Trim to exactly duration*fps. LTX returns one extra frame per chunk; keeping
        # it makes the audio slide progressively later across a long render.
        want = duration * fps
        frames = images[:want] if images.shape[0] >= want else images
        if images.shape[0] != want:
            print(f"[LTX Longform] chunk {chunk_index}: {images.shape[0]} frames "
                  f"-> keeping {frames.shape[0]} (target {want})")

        arr = (frames.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        n, h, w, _ = arr.shape

        chunk_mp4 = os.path.join(sdir, f"chunk_{chunk_index:04d}.mp4")
        # Encode to a temp name and rename. Rename is atomic on one filesystem,
        # so the finished path either does not exist or is a complete chunk -
        # which is what lets its mere existence stand in for a manifest entry.
        chunk_tmp = os.path.join(sdir, f".writing_{chunk_index:04d}.mp4")
        cmd = [ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
        # Frames go to ffmpeg on stdin, and stdin can only carry one stream, so the
        # audio has to land on disk briefly. It is deleted once muxed - the mp4
        # carries the audio from then on.
        chunk_wav = None
        if chunk_audio is not None:
            chunk_wav = os.path.join(sdir, f".chunk_{chunk_index:04d}.wav")
            self._write_wav(chunk_audio, chunk_wav, ff)
            cmd += ["-i", chunk_wav, "-map", "0:v:0", "-map", "1:a:0",
                    "-c:a", "aac", "-b:a", "192k", "-shortest"]
        else:
            cmd += ["-an"]
        cmd += ["-c:v", "libx264", "-crf", "16", "-preset", "medium",
                "-pix_fmt", "yuv420p", chunk_tmp]
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            _, err = p.communicate(arr.tobytes())
            if p.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg failed writing chunk:\n{err.decode()[-1500:]}")
        finally:
            if chunk_wav and os.path.exists(chunk_wav):
                os.remove(chunk_wav)
        os.replace(chunk_tmp, chunk_mp4)

        # Hand the last frame to the next queue item - but only if something will
        # read it. By default nothing does.
        if _anchors_needed(prompt):
            _save_anchor(frames[-1], _anchor_base(session, chunk_index))

        msg = f"chunk {chunk_index + 1}/{total_chunks} written ({n} frames)"
        print(f"[LTX Longform] {msg}")

        if chunk_index < total_chunks - 1:
            return (msg,)

        # Final chunk: stitch. Presence of the file IS the record - there is no
        # shared index to update, so several pods can write into one session
        # folder on a network volume without a write race between them.
        paths = [os.path.join(sdir, f"chunk_{i:04d}.mp4") for i in range(total_chunks)]
        missing = [i for i, p in enumerate(paths) if not os.path.exists(p)]
        if missing:
            # A gap would shift everything after it out of sync, so refuse rather
            # than quietly produce a subtly wrong video.
            raise RuntimeError(
                f"Cannot stitch: chunks {missing} are not on disk yet. If another "
                f"pod is still rendering them, wait for it and re-run this last "
                f"chunk. Otherwise queue the missing indices in session '{session}' "
                f"and run the last chunk again.")

        listfile = os.path.join(sdir, "concat.txt")
        with open(listfile, "w", encoding="utf-8") as f:
            for p in paths:
                f.write(f"file '{p}'\n")

        silent = os.path.join(sdir, "video_only.mp4")
        subprocess.run([ff, "-y", "-v", "error", "-f", "concat", "-safe", "0",
                        "-i", listfile, "-c:v", "copy", "-an", silent], check=True)

        # Mux the untouched original track. Each chunk's generated audio is a VAE
        # reconstruction of what we fed in, so the source is strictly better.
        wav_path = os.path.join(sdir, ".original.wav")
        self._write_wav(original_audio, wav_path, ff)

        out = os.path.join(folder_paths.get_output_directory(), filename)
        try:
            subprocess.run([ff, "-y", "-v", "error", "-i", silent, "-i", wav_path,
                            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                            "-c:a", "aac", "-b:a", "192k", "-shortest", out],
                           check=True)
        finally:
            # A 7-minute track as uncompressed WAV is ~74MB; it has served its
            # purpose the moment the mux completes.
            for tmp in (wav_path, silent, listfile):
                if os.path.exists(tmp):
                    os.remove(tmp)

        msg = f"FINISHED: {out}"
        print(f"[LTX Longform] {msg}")

        if stop_when_done:
            # Surplus items would be blocked by Split anyway, but clearing them
            # is tidier and makes Run (Instant) terminate instead of spinning.
            try:
                from server import PromptServer
                PromptServer.instance.prompt_queue.wipe_queue()
                print("[LTX Longform] pending queue cleared.")
            except Exception as e:
                print(f"[LTX Longform] could not clear the queue ({e}); "
                      f"surplus items will be skipped instead.")

        return (msg,)

    @staticmethod
    def _write_wav(audio, path, ff):
        wav = audio["waveform"]
        if wav.dim() == 3:
            wav = wav[0]
        arr = wav.detach().cpu().float().numpy().T  # [samples, channels]
        pcm = (arr.clip(-1, 1) * 32767).astype(np.int16)
        p = subprocess.Popen(
            [ff, "-y", "-v", "error", "-f", "s16le", "-ar",
             str(int(audio["sample_rate"])), "-ac", str(pcm.shape[1]),
             "-i", "-", path],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        _, err = p.communicate(pcm.tobytes())
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg failed writing audio:\n{err.decode()[-1500:]}")


class LTXLongformAudioName:
    """Turn the loaded audio's filename into strings for session and output name.

    Reads the selection straight off the LoadAudio node, so the file is picked once
    and the run names itself. Unlink either output and the Write node's own widget
    takes over again.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "suffix": ("STRING", {
                    "default": ".mp4",
                    "tooltip": "Appended to the filename output only; the session "
                               "name stays bare."}),
            },
            "optional": {
                "source_title": ("STRING", {
                    "default": "",
                    "tooltip": "Title of the Load Audio node to read, if the graph "
                               "has more than one. Blank uses the first found."}),
            },
            "hidden": {"prompt": "PROMPT"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("name", "filename")
    FUNCTION = "derive"
    CATEGORY = "LTX Longform"

    @classmethod
    def IS_CHANGED(cls, suffix, source_title="", prompt=None, **kw):
        # Swapping the audio file must invalidate the cache, or the previous run's
        # name would persist and the new chunks would land in the old folder.
        return f"{_audio_basename(prompt)}:{suffix}:{source_title}"

    def derive(self, suffix, source_title="", prompt=None):
        name = _audio_basename(prompt)
        print(f"[LTX Longform] run name from audio file: {name}")
        return (name, name + suffix)


NODE_CLASS_MAPPINGS = {
    "LTXLongformSplit": LTXLongformSplit,
    "LTXLongformStartFrame": LTXLongformStartFrame,
    "LTXLongformWrite": LTXLongformWrite,
    "LTXLongformAudioName": LTXLongformAudioName,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXLongformSplit": "LTX Longform: Split Audio Chunk",
    "LTXLongformStartFrame": "LTX Longform: Start Frame",
    "LTXLongformWrite": "LTX Longform: Write + Stitch",
    "LTXLongformAudioName": "LTX Longform: Name From Audio File",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
