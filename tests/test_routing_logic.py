import pytest
from router.routing_logic import classify_prompt, explain_routing_decision


def test_short_simple_prompt_routes_small():
    result = classify_prompt("Hi there", "auto", 50)
    assert result == "small"


def test_long_complex_prompt_routes_large():
    result = classify_prompt(
        "Analyze and explain in detail the transformer architecture and implement code",
        "auto",
        1024
    )
    assert result == "large"


def test_explicit_small_preference_overrides():
    result = classify_prompt(
        "Analyze and explain in detail the transformer architecture",
        "small",
        1024
    )
    assert result == "small"


def test_explicit_large_preference_overrides():
    result = classify_prompt("Hi", "large", 10)
    assert result == "large"


def test_high_token_demand_routes_large():
    result = classify_prompt("What is AI?", "auto", 600)
    assert result == "large"


def test_explain_routing_returns_dict():
    result = explain_routing_decision("Hello world", "auto", 100)
    assert "routed_to" in result
    assert "prompt_length" in result
    assert "has_complex_keyword" in result


def test_complex_keyword_detected():
    result = explain_routing_decision("Write a poem about the sea", "auto", 200)
    assert result["has_complex_keyword"] is True
