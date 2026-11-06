# m2_rewind_demo

`python scripts/make_rewind_demo.py`: our own DDIM sampling loop (`w2w.rewind.Painter`) paints each original prompt once, keeping the latent state and the predicted finished picture at every step. It then branches from steps 0, 3, 6, 9, 12 and 15 with the edited prompt. Stable Diffusion 1.5, DDIM, 30 steps, guidance 7.5. The viewer bundle is in `data/rewind_demo/` (not committed). Open `index.html` there.

| File | What it is |
|---|---|
| `demo.js` | Every number the viewer shows: prompts, words, per-step attention share, 32 x 32 attention maps per word and step (base64), edit and scene estimates, sprite layouts. 5.4 MB, most of it the attention maps. |

## Run output

```
parity: mean |pipeline - our loop| = 0.0000
[ 0.8 min] bicycle: k=0 edit 1.0 scene 0.0, k=3 edit 0.96 scene 0.356, k=6 edit 0.916 scene 0.58, k=9 edit 0.05 scene 0.851, k=12 edit 0.044 scene 0.9, k=15 edit -0.004 scene 0.931
[ 1.5 min] sunglasses: k=0 edit 1.0 scene 0.0, k=3 edit 1.034 scene 0.494, k=6 edit 0.848 scene 0.713, k=9 edit 0.912 scene 0.809, k=12 edit 0.889 scene 0.86, k=15 edit 0.94 scene 0.918
[ 2.2 min] snow: k=0 edit 1.0 scene 0.0, k=3 edit 0.973 scene 0.338, k=6 edit 0.834 scene 0.434, k=9 edit 0.796 scene 0.619, k=12 edit 0.754 scene 0.767, k=15 edit 0.726 scene 0.873
[ 2.9 min] hat: k=0 edit 1.0 scene 0.0, k=3 edit 0.931 scene 0.56, k=6 edit 1.093 scene 0.72, k=9 edit 0.025 scene 0.888, k=12 edit 0.007 scene 0.953, k=15 edit 0.001 scene 0.968
[ 3.6 min] catdog: k=0 edit 1.0 scene 0.0, k=3 edit 0.986 scene 0.117, k=6 edit 0.913 scene 0.429, k=9 edit 0.953 scene 0.675, k=12 edit 0.936 scene 0.76, k=15 edit 0.953 scene 0.83
done in 3.6 min; 16.2 MB in data/rewind_demo
```

"edit" is the CLIP estimate of how strongly the edit shows (1 = as strong as editing from the start). "scene" is the share of the original picture that survives (1 = unchanged).

Bundle size: 16.2 MB in total, made up of `demo.js` 5.4 MB, `sprites/` 7.6 MB (35 JPEGs), `finals/` 2.9 MB (35 JPEGs) and `index.html` 21 kB.

## What the viewer shows

Checked in Chromium 156 (Playwright), at 1280 x 1100 and at phone width (390 x 844).

- **Forming out of fog.** Yes. Step 1 is a colour blur, step 3 shows rough shapes, step 6 a recognisable but soft object, and from step 10 the picture is nearly finished.
- **Branches, by eye:**

  | Example | Best rewind step | What the other rewind steps do |
  |---|---|---|
  | Bicycle to motorcycle | 6: a motorcycle against the original wall | 0 and 3 add a new tree and wall; from 9 it stays a bicycle |
  | Sunglasses | 9 or 12: sunglasses on the original dog | 15 leaves only a gold, glasses-shaped blob on the ear (CLIP 0.94) |
  | Snow | 12 or 15: snow on the car and street, buildings unchanged | 3 and 6 also redraw the street |
  | Red hat | 6: the hat on the original cat | 9 and later show no hat; the scores agree |
  | Cat to dog | 9: a dog on the original bench | 12 and 15 give husky-like cat-dog hybrids, which CLIP still scores at 0.94 to 0.95 |

- **Default rewind point.** The viewer picks each example's default from the automatic scores. It chooses 6, 15, 12, 6 and 15 in the order of the table. For sunglasses and cat to dog it therefore opens on the weakest-looking branch, because CLIP is generous there.
- **Word heatmaps.**
  - They sit on the right object. "cat" covers the cat's body at step 10, and the centre of each word's overlay falls on its object.
  - The overlay is faint on bright or red regions; on the finished hat it shows only as a slight glow at the edges.
- **Page problems.**
  - Pressing Play from the finished picture holds the final frame for one tick, then starts at step 2, so step 1 is never shown. The cause is that `play()` resets the step without redrawing.
  - Clicking a rewind point briefly shows the final picture before playback starts at the rewind point.
  - The rewind buttons count steps from 0 and the step counter from 1. "Rewind to step 6" makes the first edited frame "7 of 30". That is consistent but may confuse.
  - Frames play in order. The edited and original labels switch at the right step in all 5 examples.
  - No blank frames, no console errors from the page, and no horizontal page scroll at phone width.
