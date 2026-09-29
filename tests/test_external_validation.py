import pytest
from src.external_validation import evaluate_external_juw_validation

def test_external_validation_empty():
    res = evaluate_external_juw_validation()
    assert "status" in res
    if res["status"] == "not_available":
        assert res["evaluated_count"] == 0
        assert "Real JUW external validation not available yet." in res["message"]
