# m2_rewind_live

`python -m w2w.live`: Rewind running live on this laptop's GPU (RTX 5070 Laptop, 8 GB). You type a sentence and the painting streams to the page step by step. You change the sentence and it repaints from an earlier step as a new version. The server listens on 127.0.0.1 only. Stable Diffusion 1.5, DDIM, 30 steps, guidance 7.5, and the safety checker runs on every frame.

Checked on November 3, 2026, in Chromium 156 at 1280 x 1100, on AC power.

## Timings

Seconds from pressing the button to "done", measured in the page.

| Version | What | Rewind | Seconds | Frames streamed |
|---|---|---|---|---|
| v1 | Paint new: "a cat sitting on a wooden bench", seed 1 | none | 8.65 | 30 |
| v2 | From v1: "a dog sitting on a wooden bench" | auto, after step 6 (swap) | 7.52 | 24 |
| v3 | From v2: "a dog wearing sunglasses sitting on a wooden bench" | auto, after step 9 (added detail) | 6.69 | 21 |
| v4 | From v1: "a black cat sitting on a wooden bench" | after step 3, chosen by hand | 8.78 | 27 |
| v5 | Paint new: "a red bus driving through a rainy city at night", seed 42 | none | 9.37 | 30 |
| v6 | From v5: "a yellow bus driving through a rainy city at night" | auto, after step 3 (colour) | 8.47 | 27 |

- Earlier paints of v1 took 9.17 and 9.64 s.
- The first frame arrives about 0.5 s after the button press; after that, one frame about every 0.28 s.
- A full live painting takes about 3 times as long as a plain one (3.0 s, `m1_pipeline/timing.json`), because every step is also decoded, checked and sent to the page.

## Notes

- **Frames arrive live**, step by step, in every version.
- **Branches:**
  - **v2 (dog):** the scene is kept (the same wall, bench and pose) and the cat becomes a dog.
  - **v3 (sunglasses):** the same dog in the same scene, now wearing sunglasses.
  - **v4 (black cat, from step 3):** the cat is black and white, but the right half of the scene is redrawn into a park.
  - **v6 (yellow bus, from step 3):** the bus is yellow-orange, but it is turned and moved. Starting from step 3 keeps little of the original layout.
  - Both colour edits show the colour but not the scene.
- **Auto rewind** chose step 6 for the swap, 9 for the added detail and 3 for the colour change, as described in `w2w.live`.
- **Word heatmaps:** hovering a word dims the picture and lights where that word lands. "sunglasses" lights the sunglasses and "bus" lights the bus.
- **Replay** plays the selected version from just before its rewind point. The arrow buttons step one frame at a time and stop at steps 1 and 30. At step 1 the tag reads "from v5, before the rewind".
- **The first two frames of the cat paintings are black.** The safety checker flags the blurry first two previews of "a cat sitting on a wooden bench", seed 1, which shows a harmless scene. The server log counted 26 flagged frames over the session. The bus painting's first frames were not flagged.

## Screenshots

| File | What it shows |
|---|---|
| `v1_cat.png` | The page after the first painting; the two black frames are at the start of the film strip |
| `v2_dog.png` | v2, the dog branched from v1 after step 6 |
| `v3_sunglasses_heatmap.png` | v3, with the word "sunglasses" hovered |
| `v6_bus_heatmap.png` | v6 with "bus" hovered, and all six versions |
