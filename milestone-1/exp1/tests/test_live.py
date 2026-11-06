from w2w.live import auto_rewind


def test_auto_rewind_kinds():
    base = "a cat sitting on a wooden bench"
    assert auto_rewind(base, "a dog sitting on a wooden bench") == (6, "swap")
    assert auto_rewind(base, "a cat wearing sunglasses sitting on a wooden bench") == (9, "added detail")
    assert auto_rewind(base, "a black cat sitting on a wooden bench") == (3, "colour")
    assert auto_rewind(base, "a red bus driving through a rainy city at night") == (0, "new picture")
    assert auto_rewind(base, base) == (0, "same sentence")
