# LTX Longform

Three nodes that turn one portrait + a long audio track into a minutes-long lipsync
video, using ComfyUI's batch queue as the loop. No dependencies beyond ffmpeg.

ComfyUI's graph is a DAG - each node runs once per queue item - so instead of looping
inside a node, each queue item renders one chunk and the nodes carry state between them.

## Wiring

Start from `lipsync_audio_ia2v_workflow`, then:

1. **Load Audio** (your full track) -> **LTX Longform: Split Audio Chunk**
   - `audio_chunk` -> the workflow's audio input (replacing LoadAudio)
   - `duration` -> the `Duration` input on the main node
2. **Load Image** (portrait) -> **LTX Longform: Start Frame** -> the workflow's image input
3. **VAEDecodeTiled** `IMAGE` -> **LTX Longform: Write + Stitch**
   - also connect your **full** Load Audio to `original_audio`
   - connect `total_chunks` and `duration` from the Split node

## Running

1. Queue once with `chunk_index = 0` and read `total_chunks` from the console.
2. Set `chunk_index` to **increment** (right-click the widget -> convert to input, or
   use a primitive set to increment).
3. Queue with **batch count = total_chunks**.

Chunks must run in order. The **`session`** name is set on the Write node only - Start
Frame reads it from there automatically, so there is nothing to keep in sync. Change it
to start a fresh render. The final chunk stitches everything and writes `filename` to
the output folder.

## What it handles

- **The +1 frame.** LTX renders `duration*fps + 1` frames; keeping that on every chunk
  accumulates ~2s of progressive audio slip over 7 minutes. Each chunk is written with
  exactly `duration*fps` frames.
- **Audio quality and sync.** Per-chunk audio is a VAE reconstruction of the input, so
  the final mux uses your original track instead. Sync is exact by construction.
- **Cuts land in pauses.** Boundaries are biased toward quiet stretches found in the
  waveform, so chunks do not break mid-word.
- **Identity drift.** `reanchor_every` periodically returns to the original portrait,
  trading a small visual jump for a face that still matches at the end.

## Notes

- Keep `fps` at 24. Frame counts must satisfy `frames % 8 == 1`; at 24fps every
  whole-second duration works, at 25/30/50 most durations round down and drift.
- Describe only motion and delivery in the prompt. Restating hair or clothing makes the
  model reconcile the text against the image and accelerates drift.
- If a chunk fails, re-queue that index, then run the last chunk again. Stitching
  refuses to run with gaps rather than silently shifting the timeline.
- Per-chunk files are muxed with the audio they were conditioned on, so you can check a
  single chunk for sync on its own - the fastest way to tell a generation problem from
  an assembly one.
- `cut_mode` on the Split node: `silence` cuts inside pauses (avoids breaking mid-word);
  `fixed` cuts on a grid, keeping pauses mid-chunk. Try `fixed` if speech starts early
  after a pause. The console prints how many pauses were detected.
- Intermediates live in `output/ltx_longform/<session>/`.
