import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from w2w.data import clean_caption  # noqa: E402


def test_whitespace_and_newlines():
    assert clean_caption("A man  riding\na wave .") == "A man riding a wave."


def test_leading_trailing():
    assert clean_caption("  Two dogs on a couch   ") == "Two dogs on a couch"


def test_wording_untouched():
    s = "A plate of food with brocolli and a sandwhich."
    assert clean_caption(s) == s


def test_caption_table_indexing():
    import pandas as pd
    from w2w.data import caption_table
    df = pd.DataFrame({"image_id": [10, 20], "file_name": ["a.jpg", "b.jpg"],
                       "captions": [["x one", "x two"], ["y one"]]})
    t = caption_table(df)
    assert t.caption_idx.tolist() == [0, 1, 2]
    assert t.image_id.tolist() == [10, 10, 20]
    assert t.caption.tolist() == ["x one", "x two", "y one"]


def test_recall_at_k_perfect_and_shifted():
    import numpy as np
    from w2w.retrieval import recall_at_k
    eye = np.eye(4, dtype=np.float32)
    perfect = recall_at_k(eye, eye, np.arange(4), ks=(1, 2))
    assert perfect["recall@1"] == 1.0 and perfect["median_rank"] == 1.0
    wrong = recall_at_k(eye, eye, np.array([1, 2, 3, 0]), ks=(1, 4))
    assert wrong["recall@1"] == 0.0 and wrong["recall@4"] == 1.0


def test_commit_step_and_share():
    import numpy as np
    from w2w.attention import attention_share, commit_step, is_content
    rng = np.random.default_rng(0)
    final = rng.random((64, 64))
    maps = np.zeros((10, 77, 64, 64), dtype=np.float32)
    for t in range(10):
        noise = rng.random((64, 64))
        w = 0.0 if t < 4 else 1.0            # word 1 locks into place from step 4
        maps[t, 1] = w * final + (1 - w) * noise
        maps[t, 2] = rng.random((64, 64))     # word 2 never settles
    assert commit_step(maps, 1) == 4
    assert commit_step(maps, 2) is None
    share = attention_share(maps + 1e-3, [1, 2])
    assert np.allclose(share.sum(1), 1)
    assert is_content("cat") and not is_content("the") and not is_content(".")


def test_commit_ignores_final_step_dip():
    import numpy as np
    from w2w.attention import commit_step
    rng = np.random.default_rng(1)
    final = rng.random((64, 64))
    maps = np.zeros((12, 77, 64, 64), dtype=np.float32)
    for t in range(12):
        maps[t, 1] = final if t >= 3 else rng.random((64, 64))
    maps[11, 1] = 0.5 * final + 0.5 * rng.random((64, 64))   # dip at the last evaluation
    assert commit_step(maps, 1, ignore_last=0) is None
    assert commit_step(maps, 1) == 3


def test_square_512_center_crop(tmp_path):
    from PIL import Image
    from w2w.prepare import square_512
    p = tmp_path / "wide.jpg"
    img = Image.new("L", (640, 320), 0)
    img.paste(255, (160, 0, 480, 320))       # white band exactly where the center crop lands
    img.save(p)
    out, info = square_512(p)
    assert out.size == (512, 512) and out.mode == "RGB"
    assert info["mode_in"] == "L" and info["crop_fraction_kept"] == 0.5
    px = out.getpixel((256, 256))
    assert min(px) > 240                     # the center crop kept the white band, no squash
