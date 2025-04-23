from utils.style import custom_font  # type: ignore


def test_custom_font():
    assert custom_font() != ""
