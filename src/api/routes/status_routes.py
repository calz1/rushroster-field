"""Status and health check endpoints."""

import logging
from datetime import datetime
from typing import Any

import psutil
from fastapi import FastAPI, HTTPException, Response

from ..models import StatusResponse


def register_status_routes(
    app: FastAPI,
    db: Any,
    detection_module: Any,
    startup_time: datetime,
) -> None:
    """
    Register status and health check endpoints.

    Args:
        app: FastAPI application instance
        db: Database manager instance
        detection_module: Detection module for live status (optional)
        startup_time: Application startup time for uptime calculation
    """

    @app.get("/favicon.ico")
    async def favicon():
        """Return empty response for favicon to prevent 404."""
        return Response(content="", media_type="image/x-icon")

    @app.get("/health")
    async def health_check():
        """Simple health check endpoint."""
        return {"status": "ok", "timestamp": datetime.now().isoformat()}

    @app.get("/status", response_model=StatusResponse)
    async def get_status():
        """
        Get device status including radar, camera, disk, and upload queue.
        """
        try:
            # Disk usage
            disk = psutil.disk_usage("/")
            disk_usage_percent = disk.percent
            disk_free_gb = disk.free / (1024**3)

            # Upload queue size
            unuploaded = db.get_unuploaded_events()
            upload_queue_size = len(unuploaded)

            # Network connectivity (simple check)
            network_connected = True  # TODO: Implement actual check

            # Radar/camera status
            radar_status = "Connected" if detection_module else "Unknown"
            camera_status = "Connected" if detection_module else "Unknown"

            # Uptime
            uptime = (datetime.now() - startup_time).total_seconds()

            return StatusResponse(
                radar_status=radar_status,
                camera_status=camera_status,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                upload_queue_size=upload_queue_size,
                network_connected=network_connected,
                uptime_seconds=uptime,
            )

        except Exception as e:
            logging.error(f"Error getting status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
