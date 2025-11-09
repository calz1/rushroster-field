"""API route modules."""

from .status_routes import register_status_routes
from .stats_routes import register_stats_routes
from .sensors_routes import register_sensors_routes
from .camera_routes import register_camera_routes
from .config_routes import register_config_routes
from .sync_routes import register_sync_routes
from .ui_routes import register_ui_routes
from .events_routes import register_events_routes

__all__ = [
    "register_status_routes",
    "register_stats_routes",
    "register_sensors_routes",
    "register_camera_routes",
    "register_config_routes",
    "register_sync_routes",
    "register_ui_routes",
    "register_events_routes",
]
