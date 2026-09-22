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

1. Set `chunk_index` to **0** and leave it on **increment**.
2. Queue with a batch count **at least** as large as the number of chunks. You do not
   need to know the exact number: once the render is complete, surplus queue items are
   blocked and finish in milliseconds instead of generating anything. Estimate high -
   `audio_seconds / target_seconds` rounded up, plus some margin - or just use
   **Run (Instant)**.
3. The final chunk stitches everything and, with `stop_when_done` on, clears the
   pending queue so the run ends by itself.

To start a **fresh** render: set `chunk_index` back to 0. The session name comes from
the audio filename, so loading a different file is usually all it takes.

**Resuming** an interrupted render needs no bookkeeping: queue from 0 again with
`skip_existing` on. Chunks already on disk are skipped in milliseconds and it carries
on from the gap. The final chunk always re-renders, because it is what triggers the
stitch.

## Naming runs from the audio file

`LTX Longform: Name From Audio File` reads whichever file Load Audio has selected and
outputs it two ways: `name` (bare stem, for `session`) and `filename` (stem + suffix,
for the output video). Wired up by default, so a file called `Interview_Part_02.mp3`
renders into `output/ltx_longform/Interview_Part_02/` and produces
`Interview_Part_02.mp4` with nothing to type.

Unlink either output and the Write node's own widget takes over again.

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
- **Several GPUs can share one render.** With `reanchor_every = N`, chunks 0..N-1,
  N..2N-1 and so on are independent (each group starts from the portrait), so pods
  sharing a network volume can each take a group - set `chunk_index` to that group's
  first index and use the same `session`. There is no shared index file to corrupt: a
  chunk is "done" when its .mp4 exists, and each is written to a temp name and renamed
  atomically. Run the last chunk once every other chunk is on disk; if any are still
  missing it refuses and names them.
- Per-chunk files are muxed with the audio they were conditioned on, so you can check a
  single chunk for sync on its own - the fastest way to tell a generation problem from
  an assembly one.
- `cut_mode` on the Split node:
  - **`pause`** (default) - the speech decides. `target_seconds` is ignored; it takes
    the furthest pause still within `max_seconds`, merging segments too short to
    generate. Every boundary lands in silence and chunk lengths follow the talking. A
    long unbroken sentence simply produces long chunks up to the ceiling.
  - `silence` - aims for `target_seconds`, snapping to a nearby pause if one is in
    range.
  - `fixed` - a strict grid, keeping pauses mid-chunk. Try it if speech starts early
    after a pause.

  A time cut only happens when no pause is reachable within `max_seconds` - the
  model's length ceiling forcing it, not a preference. The console prints how many
  pauses were detected; zero means every cut fell on the ceiling regardless of mode.
  A wider window finds more real pauses: on a 300s test with pauses 2-9s apart,
  `min=5 max=10` forced 10 of 42 boundaries onto the clock, while `min=3 max=12`
  forced 1 of 39.
- Intermediates live in `output/ltx_longform/<session>/`.
