from data.medial_9_forms.eval import (
    evaluate_document,
    flatten_leaves,
    indel_similarity,
)


def test_indel_similarity() -> None:
    assert indel_similarity("same", "same") == 1.0
    assert indel_similarity("a  b\n c", "a b c") == 1.0
    assert indel_similarity("", "value") == 0.0


def test_flatten_leaves_retains_array_indexes() -> None:
    assert flatten_leaves({"rows": [{"value": 1}, {"value": 2}]}) == {
        ("rows", 0, "value"): 1,
        ("rows", 1, "value"): 2,
    }


def test_missing_and_extra_fields_are_penalized() -> None:
    expected = {"name": "Alice", "age": 30}
    predicted = {"name": "Alice", "extra": True}
    metrics, _, _, denominator = evaluate_document("sample", expected, predicted)

    assert denominator == 3
    assert metrics.score == 1 / 3
    assert metrics.missing_leaves == 1
    assert metrics.extra_leaves == 1
