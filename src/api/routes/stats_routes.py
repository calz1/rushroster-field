"""Statistics and events endpoints."""

import logging
from typing import Any, Optional

from fastapi import FastAPI, HTTPException

from ..models import StatsResponse


def register_stats_routes(app: FastAPI, db: Any) -> None:
    """
    Register statistics and events endpoints.

    Args:
        app: FastAPI application instance
        db: Database manager instance
    """

    @app.get("/stats", response_model=StatsResponse)
    async def get_stats(hours: int = 24):
        """
        Get statistics for recent time period.

        Args:
            hours: Number of hours to look back (default: 24)
        """
        try:
            stats = db.get_stats(hours=hours)

            return StatsResponse(
                total_events=stats.get("total_events", 0),
                speeding_events=stats.get("speeding_events", 0),
                avg_speed=stats.get("avg_speed"),
                max_speed=stats.get("max_speed"),
                min_speed=stats.get("min_speed"),
                period_hours=hours,
            )

        except Exception as e:
            logging.error(f"Error getting stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/events")
    async def get_events(limit: int = 100, has_photos: Optional[bool] = None):
        """
        Get recent events with optional photo filter.

        Args:
            limit: Maximum number of events to return (default: 100)
            has_photos: Optional filter - True to return only events with photos,
                       False to return only events without photos,
                       None to return all events (default: None)
        """
        try:
            events = db.get_recent_events_filtered(limit=limit, has_photos=has_photos)
            return {"events": events, "count": len(events)}

        except Exception as e:
            logging.error(f"Error getting events: {e}")
            raise HTTPException(status_code=500, detail=str(e))
