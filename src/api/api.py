"""
FastAPI API for field device.
Provides status, statistics, configuration, and management endpoints.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import Config, ConfigLoader
from ..database import SQLiteManager
from .routes import (
    register_camera_routes,
    register_config_routes,
    register_events_routes,
    register_sensors_routes,
    register_stats_routes,
    register_status_routes,
    register_sync_routes,
    register_ui_routes,
)


def create_api(
    db: SQLiteManager,
    config: Config,
    config_loader: ConfigLoader,
    detection_module: Optional[Any] = None,
    camera: Optional[Any] = None,
    object_detector: Optional[Any] = None,
    dashboard_overlays: bool = False,
    sync_service: Optional[Any] = None,
) -> FastAPI:
    """
    Create FastAPI application for field device management.

    Args:
        db: Database manager instance
        config: Configuration object
        config_loader: Config loader for saving changes
        detection_module: Detection module for live status (optional)
        camera: Camera instance for live feed (optional)
        object_detector: Object detector instance for detection overlays (optional)
        dashboard_overlays: Enable object detection overlays on dashboard (optional)
        sync_service: Sync service instance for triggering manual syncs (optional)

    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="RushRoster Field Device",
        description="API for field device management and monitoring",
        version="0.1.0",
    )

    # Set up templates
    templates = Jinja2Templates(directory="src/ui/templates")

    # Track startup time for uptime calculation
    startup_time = datetime.now()

    # Register status and health check routes
    register_status_routes(
        app=app,
        db=db,
        detection_module=detection_module,
        startup_time=startup_time,
    )

    # Register statistics and events endpoints
    register_stats_routes(app=app, db=db)

    # Register sensor and streaming endpoints
    register_sensors_routes(app=app, detection_module=detection_module)

    # Register camera endpoints
    register_camera_routes(
        app=app,
        camera=camera,
        detection_module=detection_module,
        object_detector=object_detector,
        config=config,
        dashboard_overlays=dashboard_overlays,
    )

    # Register configuration endpoints
    register_config_routes(
        app=app,
        config=config,
        config_loader=config_loader,
    )

    # Register sync endpoints
    register_sync_routes(
        app=app,
        db=db,
        sync_service=sync_service,
    )

    # Register UI routes (dashboard, events browser, settings)
    register_ui_routes(
        app=app,
        config=config,
        db=db,
        templates=templates,
        detection_module=detection_module,
        config_loader=config_loader,
        startup_time=startup_time,
    )

    # Register event management routes
    register_events_routes(app=app, db=db)

    return app
