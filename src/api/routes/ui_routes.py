"""Web UI and dashboard endpoints."""

import logging
import psutil
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


def register_ui_routes(
    app: FastAPI,
    config: Any,
    db: Any,
    templates: Jinja2Templates,
    detection_module: Any,
    config_loader: Any,
    startup_time: datetime,
) -> None:
    """
    Register web UI routes (dashboard, events browser, settings).

    Args:
        app: FastAPI application instance
        config: Configuration object
        db: Database manager instance
        templates: Jinja2Templates instance
        detection_module: Detection module for live status (optional)
        config_loader: Config loader instance
        startup_time: Application startup time for uptime calculation
    """

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        """Serve diagnostic dashboard web page."""
        try:
            # Get status data
            disk = psutil.disk_usage("/")
            unuploaded = db.get_unuploaded_events()
            uptime = (datetime.now() - startup_time).total_seconds()
            radar_status = "Connected" if detection_module else "Unknown"
            camera_status = "Connected" if detection_module else "Unknown"

            status_data = {
                "radar_status": radar_status,
                "camera_status": camera_status,
                "disk_usage_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3),
                "upload_queue_size": len(unuploaded),
                "network_connected": True,
                "uptime_seconds": uptime,
            }

            # Get stats data
            stats = db.get_stats(hours=24)
            stats_data = {
                "total_events": stats.get("total_events", 0),
                "speeding_events": stats.get("speeding_events", 0),
                "avg_speed": stats.get("avg_speed"),
                "max_speed": stats.get("max_speed"),
                "min_speed": stats.get("min_speed"),
                "period_hours": 24,
            }

            # Get config data
            config_data = {
                "device": {
                    "id": config.device.id,
                    "location": {
                        "latitude": config.device.latitude,
                        "longitude": config.device.longitude,
                        "street_name": config.device.street_name,
                    },
                    "speed_limit": config.device.speed_limit,
                },
                "thresholds": {
                    "speed_threshold_mph": config.thresholds.speed_threshold_mph,
                    "photo_trigger_over_limit": config.thresholds.photo_trigger_over_limit,
                },
                "upload": {
                    "frequency_minutes": config.upload.frequency_minutes,
                },
            }

            # Get cloud config data
            cloud_config_data = {
                "api_url": config.cloud.api_url,
                "api_key": "***" + config.cloud.api_key[-8:] if config.cloud.api_key else "",
                "api_key_configured": bool(config.cloud.api_key),
            }

            return templates.TemplateResponse(
                "dashboard.html",
                {
                    "request": request,
                    "status": status_data,
                    "stats": stats_data,
                    "config": config_data,
                    "cloud_config": cloud_config_data,
                    "device_id": config.device.id,
                },
            )

        except Exception as e:
            logging.error(f"Error rendering dashboard: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/events/browse", response_class=HTMLResponse)
    async def events_browser(request: Request, page: int = 1, has_photos: Optional[bool] = None):
        """Serve events browser web page with pagination and optional photo filter."""
        try:
            # Get paginated events with optional filter
            events_data = db.get_events_paginated(page=page, page_size=20, has_photos=has_photos)

            return templates.TemplateResponse(
                "events.html",
                {
                    "request": request,
                    "events": events_data["events"],
                    "total": events_data["total"],
                    "page": events_data["page"],
                    "page_size": events_data["page_size"],
                    "total_pages": events_data["total_pages"],
                    "device_id": config.device.id,
                    "has_photos": has_photos,
                },
            )

        except Exception as e:
            logging.error(f"Error rendering events browser: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request):
        """Serve settings page."""
        try:
            # Get current configuration
            config_data = {
                "device": {
                    "id": config.device.id,
                    "location": {
                        "latitude": config.device.latitude,
                        "longitude": config.device.longitude,
                        "street_name": config.device.street_name,
                    },
                    "speed_limit": config.device.speed_limit,
                },
                "thresholds": {
                    "speed_threshold_mph": config.thresholds.speed_threshold_mph,
                    "photo_trigger_over_limit": config.thresholds.photo_trigger_over_limit,
                },
                "upload": {
                    "frequency_minutes": config.upload.frequency_minutes,
                },
            }

            # Get cloud config data
            cloud_config_data = {
                "api_url": config.cloud.api_url,
                "api_key": "***" + config.cloud.api_key[-8:] if config.cloud.api_key else "",
                "api_key_configured": bool(config.cloud.api_key),
            }

            return templates.TemplateResponse(
                "settings.html",
                {
                    "request": request,
                    "config": config_data,
                    "cloud_config": cloud_config_data,
                    "device_id": config.device.id,
                },
            )

        except Exception as e:
            logging.error(f"Error rendering settings page: {e}")
            raise HTTPException(status_code=500, detail=str(e))
