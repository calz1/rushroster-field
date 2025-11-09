"""Live sensor data and streaming endpoints."""

import asyncio
import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse


def register_sensors_routes(app: FastAPI, detection_module: Any) -> None:
    """
    Register live sensor and streaming endpoints.

    Args:
        app: FastAPI application instance
        detection_module: Detection module for live status (optional)
    """

    @app.get("/sensors/live")
    async def get_live_sensors():
        """
        Get live sensor readings (radar speed, camera status, tracking state).
        """
        try:
            live_data = {
                "radar": {
                    "current_speed": None,
                    "tracking": False,
                    "target_acquired": False,
                },
                "camera": {
                    "active": False,
                    "frames_captured": 0,
                },
                "detection": {
                    "state": "idle",
                    "max_speed_this_target": 0,
                },
            }

            # Get detection module stats if available
            if detection_module:
                stats = detection_module.get_statistics()
                live_data["radar"]["tracking"] = stats.get("tracking", False)
                live_data["radar"]["target_acquired"] = stats.get("target_acquired", False)
                live_data["radar"]["current_speed"] = stats.get("recent_speed", 0)
                live_data["detection"]["state"] = "tracking" if stats.get("tracking") else "idle"

                # Get max speed for current target
                if hasattr(detection_module, 'max_speed_seen'):
                    live_data["detection"]["max_speed_this_target"] = abs(detection_module.max_speed_seen)

            return live_data

        except Exception as e:
            logging.error(f"Error getting live sensor data: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/sensors/stream")
    async def stream_sensors():
        """
        Server-Sent Events stream for live sensor data.
        More efficient than polling - maintains single connection.
        """
        async def event_generator():
            try:
                while True:
                    # Get live sensor data
                    live_data = {
                        "radar": {
                            "current_speed": None,
                            "tracking": False,
                            "target_acquired": False,
                        },
                        "detection": {
                            "state": "idle",
                            "max_speed_this_target": 0,
                        },
                    }

                    if detection_module:
                        stats = detection_module.get_statistics()
                        live_data["radar"]["tracking"] = stats.get("tracking", False)
                        live_data["radar"]["target_acquired"] = stats.get("target_acquired", False)
                        live_data["radar"]["current_speed"] = stats.get("recent_speed", 0)
                        live_data["detection"]["state"] = "tracking" if stats.get("tracking") else "idle"

                        if hasattr(detection_module, 'max_speed_seen'):
                            live_data["detection"]["max_speed_this_target"] = abs(detection_module.max_speed_seen)

                    # Send as SSE event
                    yield f"data: {json.dumps(live_data)}\n\n"

                    # Wait 200ms before next update (5 updates/sec)
                    await asyncio.sleep(0.2)

            except asyncio.CancelledError:
                # Client disconnected
                pass

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            }
        )
