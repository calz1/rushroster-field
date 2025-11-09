"""Configuration endpoints."""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException

from ..models import CloudConfigUpdateRequest, ConfigUpdateRequest


def register_config_routes(
    app: FastAPI,
    config: Any,
    config_loader: Any,
) -> None:
    """
    Register configuration endpoints.

    Args:
        app: FastAPI application instance
        config: Configuration object
        config_loader: Config loader for saving changes
    """

    @app.get("/config")
    async def get_config():
        """Get current device configuration."""
        try:
            return {
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

        except Exception as e:
            logging.error(f"Error getting config: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.put("/config")
    async def update_config(request: ConfigUpdateRequest):
        """
        Update device configuration.

        Args:
            request: Configuration update request
        """
        try:
            updated = False

            if request.speed_limit is not None:
                config.device.speed_limit = request.speed_limit
                updated = True

            if request.photo_trigger_over_limit is not None:
                config.thresholds.photo_trigger_over_limit = (
                    request.photo_trigger_over_limit
                )
                updated = True

            if request.upload_frequency_minutes is not None:
                config.upload.frequency_minutes = request.upload_frequency_minutes
                updated = True

            if updated:
                # Save config to file
                config_loader.save(config)
                logging.info("Configuration updated")

            return {"status": "ok", "message": "Configuration updated"}

        except Exception as e:
            logging.error(f"Error updating config: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/config/cloud")
    async def get_cloud_config():
        """Get cloud configuration (API key is masked for security)."""
        try:
            return {
                "api_url": config.cloud.api_url,
                "api_key": "***" + config.cloud.api_key[-8:] if config.cloud.api_key else "",
                "api_key_configured": bool(config.cloud.api_key),
            }

        except Exception as e:
            logging.error(f"Error getting cloud config: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.put("/config/cloud")
    async def update_cloud_config(request: CloudConfigUpdateRequest):
        """
        Update cloud configuration.

        Args:
            request: Cloud configuration update request
        """
        try:
            updated = False

            if request.api_url is not None:
                config.cloud.api_url = request.api_url.rstrip("/")
                updated = True
                logging.info(f"Updated cloud API URL to: {config.cloud.api_url}")

            if request.api_key is not None:
                config.cloud.api_key = request.api_key
                updated = True
                logging.info("Updated cloud API key")

            if updated:
                # Save config to file
                config_loader.save(config)
                logging.info("Cloud configuration saved")

            return {"status": "ok", "message": "Cloud configuration updated"}

        except Exception as e:
            logging.error(f"Error updating cloud config: {e}")
            raise HTTPException(status_code=500, detail=str(e))
