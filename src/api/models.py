"""Pydantic models for API requests and responses."""

from typing import Optional
from pydantic import BaseModel


class StatusResponse(BaseModel):
    """Device status response."""

    radar_status: str
    camera_status: str
    disk_usage_percent: float
    disk_free_gb: float
    upload_queue_size: int
    network_connected: bool
    uptime_seconds: float


class StatsResponse(BaseModel):
    """Statistics response."""

    total_events: int
    speeding_events: int
    avg_speed: Optional[float]
    max_speed: Optional[float]
    min_speed: Optional[float]
    period_hours: int


class EventResponse(BaseModel):
    """Event response."""

    id: int
    timestamp: str
    speed: float
    speed_limit: float
    is_speeding: bool
    photo_path: Optional[str]
    uploaded: bool


class ConfigUpdateRequest(BaseModel):
    """Configuration update request."""

    speed_limit: Optional[float] = None
    photo_trigger_over_limit: Optional[float] = None
    upload_frequency_minutes: Optional[int] = None


class CloudConfigUpdateRequest(BaseModel):
    """Cloud configuration update request."""

    api_url: Optional[str] = None
    api_key: Optional[str] = None
