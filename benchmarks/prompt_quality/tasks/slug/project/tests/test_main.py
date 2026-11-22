from main import make_slug


def test_ascii_words_and_whitespace():
    assert make_slug("  FastAPI   prompt tools ") == "fastapi-prompt-tools"


def test_punctuation_and_repeated_separators():
    assert make_slug("FastAPI, Python & AI") == "fastapi-python-ai"
    assert make_slug("hello---world___again") == "hello-world-again"


def test_unicode_letters_and_casefolding():
    assert make_slug("Café déjà STRAßE 42") == "café-déjà-strasse-42"


def test_non_alphanumeric_input_is_empty():
    assert make_slug(" -- !! ") == ""
