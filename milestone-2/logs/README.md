# Milestone 2 logs

| File | What it holds |
|---|---|
| `images_all.csv` | every image any of us generated for Milestone 2, one row each (23,673 rows): who made it, model, setting, sampler, schedule, steps, guidance, prompt set, caption, seed id, seed, seed rule, blacked out or not, seconds, GPU, file |
| `runs_all.csv` | the same images grouped into 47 runs: one row per member, model, setting and prompt set, with image count, blacked-out count and seconds per image |
| `runs_all.md` | `runs_all.csv` as a table (Appendix A of the report) |
| `training_log_connector.csv` | the BERT connector's stage-1 training log (Appendix B): training loss every 50 steps, held-out loss and cosine to CLIP's states every 250 steps, minutes and learning rate; the original is in `../exp2/` |

`images_all.csv` and `runs_all.*` are built by `exp1/scripts/m2_merge_logs.py` from each member's `runs.csv` and `images.csv` (or, for runs made before the shared protocol, their own generation log). Per member: Santhoshkumar 9,953 images, Tanmayi 6,600, Mukul 4,920, Praniti 2,200; 96 were blacked out by the safety checker and left out of every score.
