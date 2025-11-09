"""
SQLite database manager for local event storage on field device.
Implements the schema defined in the technical specification.
"""

import sqlite3
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from pathlib import Path


class SQLiteManager:
    """Manages local SQLite database for speed events on field device."""

    def __init__(self, db_path: str = "data/events.db"):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_directory()

    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.

        Yields:
            sqlite3.Connection: Database connection
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()

    def initialize_database(self):
        """Create database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Events table - detection events
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    speed REAL NOT NULL,
                    speed_limit REAL NOT NULL,
                    is_speeding BOOLEAN NOT NULL,
                    photo_path TEXT,
                    uploaded BOOLEAN DEFAULT 0,
                    uploaded_at DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Device configuration table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS device_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Device metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS device_metadata (
                    device_id TEXT PRIMARY KEY,
                    latitude REAL,
                    longitude REAL,
                    street_name TEXT,
                    speed_limit REAL,
                    last_sync DATETIME
                )
            """)

            # Create indices for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp
                ON events(timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_uploaded
                ON events(uploaded, timestamp)
            """)

            conn.commit()
            logging.info(f"Database initialized: {self.db_path}")

    def store_event(
        self,
        speed: float,
        speed_limit: float,
        is_speeding: bool,
        photo_path: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> int:
        """
        Store a speed detection event.

        Args:
            speed: Detected speed (MPH)
            speed_limit: Current speed limit (MPH)
            is_speeding: Whether vehicle was speeding
            photo_path: Path to captured photo (if any)
            timestamp: Event timestamp (defaults to now)

        Returns:
            ID of inserted event
        """
        if timestamp is None:
            timestamp = datetime.now()

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO events (timestamp, speed, speed_limit, is_speeding, photo_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (timestamp.isoformat(), speed, speed_limit, is_speeding, photo_path),
            )
            event_id = cursor.lastrowid
            conn.commit()

            logging.debug(
                f"Event stored: ID={event_id}, speed={speed}, "
                f"is_speeding={is_speeding}, photo={photo_path}"
            )
            return event_id

    def update_event_photo(self, event_id: int, photo_path: str):
        """
        Update the photo path for an existing event.
        Used by burst capture to update the event after async processing completes.

        Args:
            event_id: Event ID to update
            photo_path: Path to the selected best frame
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE events
                SET photo_path = ?
                WHERE id = ?
                """,
                (photo_path, event_id)
            )

            conn.commit()
            logging.debug(f"Updated event {event_id} with photo: {photo_path}")

    def get_unuploaded_events(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get events that haven't been uploaded to cloud.

        Args:
            limit: Maximum number of events to retrieve

        Returns:
            List of event dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT * FROM events
                WHERE uploaded = 0
                ORDER BY timestamp ASC
            """

            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def mark_events_uploaded(self, event_ids: List[int]):
        """
        Mark events as uploaded to cloud.

        Args:
            event_ids: List of event IDs to mark as uploaded
        """
        if not event_ids:
            return

        with self.get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(event_ids))
            cursor.execute(
                f"""
                UPDATE events
                SET uploaded = 1, uploaded_at = ?
                WHERE id IN ({placeholders})
                """,
                [datetime.now().isoformat()] + event_ids,
            )
            conn.commit()
            logging.info(f"Marked {len(event_ids)} events as uploaded")

    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent events for display/diagnostics.

        Args:
            limit: Maximum number of events to retrieve

        Returns:
            List of event dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM events
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_events_filtered(
        self, limit: int = 100, has_photos: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent events with optional filtering by photo presence.

        Args:
            limit: Maximum number of events to retrieve
            has_photos: If True, return only events with photos. If False, return only events without photos.
                       If None, return all events regardless of photo presence.

        Returns:
            List of event dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM events"
            params = []

            # Add photo filter if specified
            if has_photos is not None:
                if has_photos:
                    query += " WHERE photo_path IS NOT NULL"
                else:
                    query += " WHERE photo_path IS NULL"

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_stats(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get statistics for recent time period.

        Args:
            hours: Number of hours to look back

        Returns:
            Dictionary with statistics
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Total events in period
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total_events,
                    COUNT(CASE WHEN is_speeding = 1 THEN 1 END) as speeding_events,
                    AVG(speed) as avg_speed,
                    MAX(speed) as max_speed,
                    MIN(speed) as min_speed
                FROM events
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
                """,
                (hours,),
            )

            row = cursor.fetchone()
            return dict(row) if row else {}

    def set_config(self, key: str, value: str):
        """
        Set a configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO device_config (key, value, updated_at)
                VALUES (?, ?, ?)
                """,
                (key, value, datetime.now().isoformat()),
            )
            conn.commit()
            logging.debug(f"Config set: {key}={value}")

    def get_config(self, key: str) -> Optional[str]:
        """
        Get a configuration value.

        Args:
            key: Configuration key

        Returns:
            Configuration value or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT value FROM device_config WHERE key = ?
                """,
                (key,),
            )
            row = cursor.fetchone()
            return row["value"] if row else None

    def get_all_config(self) -> Dict[str, str]:
        """
        Get all configuration values.

        Returns:
            Dictionary of all configuration key-value pairs
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM device_config")
            return {row["key"]: row["value"] for row in cursor.fetchall()}

    def update_metadata(
        self,
        device_id: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        street_name: Optional[str] = None,
        speed_limit: Optional[float] = None,
    ):
        """
        Update device metadata.

        Args:
            device_id: Unique device identifier
            latitude: Device latitude
            longitude: Device longitude
            street_name: Street name where device is located
            speed_limit: Speed limit at device location
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Check if metadata exists
            cursor.execute(
                "SELECT device_id FROM device_metadata WHERE device_id = ?",
                (device_id,),
            )
            exists = cursor.fetchone() is not None

            if exists:
                # Update existing
                updates = []
                values = []

                if latitude is not None:
                    updates.append("latitude = ?")
                    values.append(latitude)
                if longitude is not None:
                    updates.append("longitude = ?")
                    values.append(longitude)
                if street_name is not None:
                    updates.append("street_name = ?")
                    values.append(street_name)
                if speed_limit is not None:
                    updates.append("speed_limit = ?")
                    values.append(speed_limit)

                updates.append("last_sync = ?")
                values.append(datetime.now().isoformat())
                values.append(device_id)

                if updates:
                    cursor.execute(
                        f"""
                        UPDATE device_metadata
                        SET {', '.join(updates)}
                        WHERE device_id = ?
                        """,
                        values,
                    )
            else:
                # Insert new
                cursor.execute(
                    """
                    INSERT INTO device_metadata
                    (device_id, latitude, longitude, street_name, speed_limit, last_sync)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        device_id,
                        latitude,
                        longitude,
                        street_name,
                        speed_limit,
                        datetime.now().isoformat(),
                    ),
                )

            conn.commit()
            logging.info(f"Metadata updated for device {device_id}")

    def get_metadata(self, device_id: str) -> Optional[Dict[str, Any]]:
        """
        Get device metadata.

        Args:
            device_id: Unique device identifier

        Returns:
            Metadata dictionary or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM device_metadata WHERE device_id = ?", (device_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_events_paginated(
        self, page: int = 1, page_size: int = 20, has_photos: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Get events with pagination support and optional photo filter.

        Args:
            page: Page number (1-indexed)
            page_size: Number of events per page
            has_photos: If True, return only events with photos. If False, return only events without photos.
                       If None, return all events regardless of photo presence.

        Returns:
            Dictionary with events list, total count, and pagination info
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Build query with optional filter
            where_clause = ""
            if has_photos is not None:
                if has_photos:
                    where_clause = "WHERE photo_path IS NOT NULL"
                else:
                    where_clause = "WHERE photo_path IS NULL"

            # Get total count with filter
            count_query = f"SELECT COUNT(*) as total FROM events {where_clause}"
            cursor.execute(count_query)
            total = cursor.fetchone()["total"]

            # Get paginated events with filter
            offset = (page - 1) * page_size
            query = f"""
                SELECT * FROM events
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """
            cursor.execute(query, (page_size, offset))
            events = [dict(row) for row in cursor.fetchall()]

            return {
                "events": events,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size,
            }

    def delete_events(self, event_ids: List[int]) -> int:
        """
        Delete specific events by ID.

        Args:
            event_ids: List of event IDs to delete

        Returns:
            Number of events deleted
        """
        if not event_ids:
            return 0

        with self.get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(event_ids))
            cursor.execute(
                f"""
                DELETE FROM events
                WHERE id IN ({placeholders})
                """,
                event_ids,
            )
            deleted_count = cursor.rowcount
            conn.commit()
            logging.info(f"Deleted {deleted_count} events")
            return deleted_count

    def delete_all_events(self) -> int:
        """
        Delete all events from the database.

        Returns:
            Number of events deleted
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM events")
            deleted_count = cursor.rowcount
            conn.commit()
            logging.info(f"Deleted all {deleted_count} events")
            return deleted_count

    def get_total_event_count(self) -> int:
        """
        Get total count of all events in the database.

        Returns:
            Total number of events
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM events")
            row = cursor.fetchone()
            return row["count"] if row else 0

    def reset_all_upload_status(self):
        """
        Reset all events to not uploaded status, forcing a complete re-sync.
        Sets uploaded = 0 and uploaded_at = NULL for all events.
        Useful after photo upload errors or cloud sync issues.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE events
                SET uploaded = 0, uploaded_at = NULL
                WHERE uploaded = 1
                """
            )
            updated_count = cursor.rowcount
            conn.commit()
            logging.info(f"Reset upload status for {updated_count} events")
            return updated_count
