from cluster_count.data import parse_count_from_filename


def test_parse_explicit_count_token() -> None:
    assert parse_count_from_filename("sample_count_42.png") == 42


def test_parse_simcep_style_c_token() -> None:
    assert parse_count_from_filename("SIMCEPImages_A06_C23_F1_s11_w2.TIF") == 23


def test_parse_trailing_integer() -> None:
    assert parse_count_from_filename("normal_17.tif") == 17

