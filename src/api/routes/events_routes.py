"""Event management endpoints (view, delete, serve photos)."""

import logging
from pathlib import Path
from typing import Any, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response


def register_events_routes(app: FastAPI, db: Any) -> None:
    """
    Register event management endpoints.

    Args:
        app: FastAPI application instance
        db: Database manager instance
    """

    @app.post("/events/delete")
    async def delete_events_endpoint(event_ids: List[int]):
        """
        Delete specific events by ID.

        Args:
            event_ids: List of event IDs to delete
        """
        try:
            deleted_count = db.delete_events(event_ids)
            return {
                "status": "ok",
                "message": f"Deleted {deleted_count} event(s)",
                "deleted_count": deleted_count,
            }

        except Exception as e:
            logging.error(f"Error deleting events: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/events/clear")
    async def clear_all_events():
        """Delete all events from the database."""
        try:
            deleted_count = db.delete_all_events()
            return {
                "status": "ok",
                "message": f"Cleared all events ({deleted_count} deleted)",
                "deleted_count": deleted_count,
            }

        except Exception as e:
            logging.error(f"Error clearing events: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/events/photo/{event_id}")
    async def get_event_photo(event_id: int):
        """
        Serve photo for a specific event.

        Args:
            event_id: Event ID
        """
        try:
            # Get event from database to find photo path
            events = db.get_recent_events(limit=10000)  # Get all events
            event = next((e for e in events if e["id"] == event_id), None)

            if not event:
                raise HTTPException(status_code=404, detail="Event not found")

            photo_path = event.get("photo_path")
            if not photo_path:
                raise HTTPException(status_code=404, detail="No photo for this event")

            # Check if file exists - handle both absolute and relative paths
            photo_file = Path(photo_path)
            if not photo_file.is_absolute():
                # For relative paths, resolve them from the current working directory
                photo_file = photo_file.resolve()

            if not photo_file.exists():
                raise HTTPException(status_code=404, detail="Photo file not found")

            # Read and return the image file
            with open(photo_file, "rb") as f:
                image_data = f.read()

            # Determine content type based on file extension
            content_type = "image/jpeg"
            if str(photo_file).lower().endswith(".png"):
                content_type = "image/png"

            return Response(content=image_data, media_type=content_type)

        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Error serving photo: {e}")
            raise HTTPException(status_code=500, detail=str(e))
