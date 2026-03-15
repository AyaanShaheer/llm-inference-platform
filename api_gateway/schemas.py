from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
import uuid


class InferenceRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = Field(..., min_length=1, max_length=4096)
    model_preference: Optional[Literal["small", "large", "auto"]] = "auto"
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be blank or whitespace only")
        return v.strip()


class InferenceResponse(BaseModel):
    request_id: str
    model_used: str
    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    status: Literal["success", "error"] = "success"


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str = "1.0.0"
    router_reachable: bool = False


class ErrorResponse(BaseModel):
    request_id: str
    error: str
    status: str = "error"
