import pytest
from pydantic import ValidationError
from api_gateway.schemas import InferenceRequest


def test_valid_request():
    req = InferenceRequest(prompt="Hello world", max_tokens=100)
    assert req.prompt == "Hello world"
    assert req.model_preference == "auto"
    assert req.request_id is not None


def test_blank_prompt_rejected():
    with pytest.raises(ValidationError):
        InferenceRequest(prompt="   ", max_tokens=100)


def test_empty_prompt_rejected():
    with pytest.raises(ValidationError):
        InferenceRequest(prompt="", max_tokens=100)


def test_max_tokens_bounds():
    with pytest.raises(ValidationError):
        InferenceRequest(prompt="test", max_tokens=9999)
    with pytest.raises(ValidationError):
        InferenceRequest(prompt="test", max_tokens=0)


def test_temperature_bounds():
    with pytest.raises(ValidationError):
        InferenceRequest(prompt="test", temperature=3.0)


def test_model_preference_validation():
    req = InferenceRequest(prompt="test", model_preference="large")
    assert req.model_preference == "large"
    with pytest.raises(ValidationError):
        InferenceRequest(prompt="test", model_preference="xlarge")
