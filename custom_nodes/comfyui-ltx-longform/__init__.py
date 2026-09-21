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

import json
import os
import shutil
import subprocess

import numpy as np
import torch

import folder_paths

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


def plan_chunks(total, target, min_len, max_len, pauses):
    """Split [0,total] into whole-second spans, preferring cuts at pauses.

    Whole seconds matter: the workflow computes frames as duration*fps + 1, and
    ComfyUI floor-divides that back into latent frames, so a fractional duration
    silently yields a shorter clip than asked for.
    """
    spans, pos = [], 0.0
    while total - pos > max_len:
        ideal = pos + target
        lo, hi = pos + min_len, pos + max_len
        cands = [p for p in pauses if lo <= p <= hi]
        cut = min(cands, key=lambda p: abs(p - ideal)) if cands else ideal
        cut = pos + max(min_len, min(max_len, round(cut - pos)))
        if cut <= pos:
            cut = pos + target
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
            }
        }

    RETURN_TYPES = ("AUDIO", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("audio_chunk", "duration", "total_chunks", "is_last")
    FUNCTION = "split"
    CATEGORY = "LTX Longform"

    def split(self, audio, chunk_index, target_seconds, min_seconds, max_seconds):
        if min_seconds > max_seconds:
            min_seconds, max_seconds = max_seconds, min_seconds
        target_seconds = max(min_seconds, min(max_seconds, target_seconds))

        wav, sr = audio["waveform"], int(audio["sample_rate"])
        total = wav.shape[-1] / sr
        spans = plan_chunks(total, target_seconds, min_seconds, max_seconds,
                            _find_pauses(_mono(wav), sr))
        if not spans:
            raise RuntimeError("Audio too short to split into chunks.")

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
                "session": ("STRING", {"default": "run1"}),
                "reanchor_every": ("INT", {"default": 6, "min": 0, "max": 1000,
                                           "tooltip": "0 = never re-anchor."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("start_frame",)
    FUNCTION = "pick"
    CATEGORY = "LTX Longform"

    @classmethod
    def IS_CHANGED(cls, chunk_index, session, **kw):
        # The file on disk changes between queue items while the inputs may not, so
        # report the previous chunk's frame as part of the cache key.
        p = os.path.join(_session_dir(session), f"last_{chunk_index - 1:04d}.npy")
        return f"{chunk_index}:{os.path.getmtime(p) if os.path.exists(p) else 0}"

    def pick(self, portrait, chunk_index, session, reanchor_every):
        if chunk_index == 0:
            print("[LTX Longform] chunk 0: starting from the portrait")
            return (portrait,)
        if reanchor_every and chunk_index % reanchor_every == 0:
            print(f"[LTX Longform] chunk {chunk_index}: re-anchoring to the portrait")
            return (portrait,)

        prev = os.path.join(_session_dir(session), f"last_{chunk_index - 1:04d}.npy")
        if not os.path.exists(prev):
            print(f"[LTX Longform] WARNING: no last frame from chunk "
                  f"{chunk_index - 1}; falling back to the portrait. "
                  f"(Queue chunks in order, and keep the session name the same.)")
            return (portrait,)
        arr = np.load(prev)
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
                "session": ("STRING", {"default": "run1"}),
                "filename": ("STRING", {"default": "longform_final.mp4"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "write"
    OUTPUT_NODE = True
    CATEGORY = "LTX Longform"

    def write(self, images, original_audio, chunk_index, total_chunks,
              duration, fps, session, filename):
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
        p = subprocess.Popen(
            [ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
             "-an", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
             "-pix_fmt", "yuv420p", chunk_mp4],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        _, err = p.communicate(arr.tobytes())
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg failed writing chunk:\n{err.decode()[-1500:]}")

        # Hand the last frame to the next queue item.
        np.save(os.path.join(sdir, f"last_{chunk_index:04d}.npy"),
                frames[-1].detach().cpu().numpy())

        manifest = os.path.join(sdir, "manifest.json")
        done = json.load(open(manifest)) if os.path.exists(manifest) else {}
        done[str(chunk_index)] = os.path.basename(chunk_mp4)
        json.dump(done, open(manifest, "w"), indent=1)

        msg = f"chunk {chunk_index + 1}/{total_chunks} written ({n} frames)"
        print(f"[LTX Longform] {msg}")

        if chunk_index < total_chunks - 1:
            return (msg,)

        # Final chunk: stitch. Missing chunks are reported rather than silently
        # skipped, since a gap would shift everything after it out of sync.
        missing = [i for i in range(total_chunks) if str(i) not in done]
        if missing:
            raise RuntimeError(
                f"Cannot stitch: chunks {missing} were never rendered. Re-queue them "
                f"with the same session name, then run the last chunk again.")

        listfile = os.path.join(sdir, "concat.txt")
        with open(listfile, "w", encoding="utf-8") as f:
            for i in range(total_chunks):
                f.write(f"file '{os.path.join(sdir, done[str(i)])}'\n")

        silent = os.path.join(sdir, "video_only.mp4")
        subprocess.run([ff, "-y", "-v", "error", "-f", "concat", "-safe", "0",
                        "-i", listfile, "-c", "copy", silent], check=True)

        # Mux the untouched original track. Each chunk's generated audio is a VAE
        # reconstruction of what we fed in, so the source is strictly better.
        wav_path = os.path.join(sdir, "original.wav")
        self._write_wav(original_audio, wav_path, ff)

        out = os.path.join(folder_paths.get_output_directory(), filename)
        subprocess.run([ff, "-y", "-v", "error", "-i", silent, "-i", wav_path,
                        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=True)

        msg = f"FINISHED: {out}"
        print(f"[LTX Longform] {msg}")
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


NODE_CLASS_MAPPINGS = {
    "LTXLongformSplit": LTXLongformSplit,
    "LTXLongformStartFrame": LTXLongformStartFrame,
    "LTXLongformWrite": LTXLongformWrite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXLongformSplit": "LTX Longform: Split Audio Chunk",
    "LTXLongformStartFrame": "LTX Longform: Start Frame",
    "LTXLongformWrite": "LTX Longform: Write + Stitch",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
