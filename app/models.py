from pydantic import BaseModel, Field
from datetime import datetime, timezone


class ChatRequest(BaseModel):
    """Request model for chat messages."""

    message: str = Field(..., min_length=1, max_length=10000,description="The message content from the user.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the message.")

class ChatResponse(BaseModel):
    """Response model for chat messages."""

    response : str
    thread_id: str
    model_used: str
    cached : bool
    processing_time: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of the response.")


class HealthCheckResponse(BaseModel):
    """Response model for health check."""

    status: str = "Healty"
    envirnoment : str
    version : str = "1.0.0"
    checks : dict = {}

class MetricsResponse(BaseModel):
    """Response model for metrics."""

    total_requests: int
    total_error: int
    error_rate: str
    avg_latency_ms: float
    cache_hit_rate: str
    total_input_tokens: int
    total_output_tokens: int


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str
    details: str = None
    request_id: str = None