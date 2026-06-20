from dual_track_opd.eval.answer_extractors import (
    extract_choice,
    normalize_numeric_answer,
    normalize_text_answer,
)


def test_extract_choice_accepts_explicit_formats():
    assert extract_choice("Answer: C") == "C"
    assert extract_choice("(C)") == "C"
    assert extract_choice("C.") == "C"
    assert extract_choice("C. The object is on the left.") == "C"
    assert extract_choice("The answer is C because the image shows it.") == "C"


def test_extract_choice_rejects_ambiguous_long_text():
    assert extract_choice("A cat is sitting beside a cup.") is None
    assert extract_choice("It could be A or B.") is None


def test_normalize_text_answer():
    assert normalize_text_answer("The Red, Apple!") == "red apple"
    assert normalize_text_answer("  An   object.  ") == "object"


def test_normalize_numeric_answer():
    assert normalize_numeric_answer("Answer: 3.0") == "3"
    assert normalize_numeric_answer("1,024") == "1024"
    assert normalize_numeric_answer("values are 3 and 4") is None
