# Human check of the object judges

Every result in Milestone 2 depends on two automatic judges (Faster R-CNN v2 and OWL-ViT). Here the four of us checked them by eye.

**Design.** 240 development pictures (from the early-warning study's development paintings), 80 from each judge group: both judges miss the object, one finds it, both find it. Each picture was labelled "yes", "no" or "can't tell" by two of us, without seeing the judges' calls, on `picture-check.html`. Rule given to raters: answer for the exact object named; a truck, bus or van is not a car.

**Results.**

| Measure | Value |
|---|---|
| Agreement between raters, when both decided (165 pictures) | 95.2%, kappa 0.90 |
| Pictures with a clear human verdict | 216 of 240 |
| Our rule's "missing" is really missing (weighted to all development objects) | 97.3% |
| Our rule's overall accuracy | 91.7% [86.5, 96.0] |
| Objects only one judge finds that people call missing | 39% |
| Faster R-CNN score on the finished picture, AUROC against people | 0.96 |
| RetinaNet score on the finished picture, AUROC against people | 0.92 |

The rule's "missing" calls are reliable, but it misses some real losses, so the missing rates in the report are lower bounds.

## Files

| File | What it holds |
|---|---|
| `audit_sample.csv` | the 240 pictures: item id, seed id, caption, object, judge scores, judge group |
| `group_shares.json` | each judge group's share of all development objects, used to weight the results |
| `answers_*.txt` | each rater's answers, one code per picture (`p000y` = yes, `n` = no, `u` = can't tell) |
| `audit_verdicts.csv` | the combined human verdict for every picture |
| `audit_analyze.py` | recomputes every number above: `python audit_analyze.py .` |
| `picture-check.html` | the labelling page; open it in a browser with the pictures in `img/` |
| `img/` | the 240 pictures shown to raters (`p000.jpg` to `p239.jpg`) |
