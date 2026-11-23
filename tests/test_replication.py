import pytest

from scripts.replicate_stage1 import summarize


def test_replication_reports_every_seed_and_rejects_missing_or_duplicate_results():
    def row(seed, accuracy):
        metric = {"exact_vector_accuracy": accuracy, "per_rule": [
            {"recall": accuracy, "balanced_accuracy": accuracy} for _ in range(8)]}
        return {"seed": seed, "sample": metric, "contrast": metric}
    records = [row(106, 0.9), row(107, 0.8)]
    result = summarize(records, [106, 107])
    assert result["contrast"]["exact_accuracy_mean"] == pytest.approx(0.85)
    assert result["contrast"]["exact_accuracy_min"] == 0.8
    assert [r["seed"] for r in result["per_seed"]] == [106, 107]
    assert result["contrast"]["per_rule_recall_min"] == [0.8] * 8
    with pytest.raises(ValueError, match="every"):
        summarize(records[:1], [106, 107])
    with pytest.raises(ValueError, match="every"):
        summarize([records[0], records[0]], [106, 107])
