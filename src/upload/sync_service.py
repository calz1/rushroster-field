"""
Data upload service for field device.
Periodically uploads events and photos to cloud platform.

NOTE: This is currently a stub implementation.
Full implementation will be added when cloud platform is ready.
"""

import time
import logging
import threading
import requests
from typing import Optional
from datetime import datetime

from ..database import SQLiteManager
from ..config import CloudConfig


class SyncService:
    """Background service for uploading data to cloud platform."""

    def __init__(
        self,
        db: SQLiteManager,
        cloud_config: CloudConfig,
        frequency_minutes: int = 15,
        batch_size: int = 100,
        retry_attempts: int = 3,
    ):
        """
        Initialize upload service.

        Args:
            db: Database manager instance
            cloud_config: Cloud configuration
            frequency_minutes: How often to upload (minutes)
            batch_size: Max events per batch
            retry_attempts: Number of retry attempts for failed uploads
        """
        self.db = db
        self.cloud_config = cloud_config
        self.frequency_minutes = frequency_minutes
        self.batch_size = batch_size
        self.retry_attempts = retry_attempts

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Statistics
        self.last_upload_time: Optional[datetime] = None
        self.total_uploaded = 0
        self.upload_failures = 0

    def start(self):
        """Start the upload service in background thread."""
        if self._running:
            logging.warning("Upload service already running")
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logging.info(
            f"Upload service started (frequency: {self.frequency_minutes} minutes)"
        )

    def stop(self):
        """Stop the upload service."""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=5.0)

        logging.info("Upload service stopped")

    def _run_loop(self):
        """Main upload loop that runs in background thread."""
        while self._running:
            try:
                # Wait for next upload interval
                if self._stop_event.wait(timeout=self.frequency_minutes * 60):
                    # Stop event was set
                    break

                # Perform upload
                self._upload()

            except Exception as e:
                logging.error(f"Error in upload loop: {e}", exc_info=True)
                self.upload_failures += 1

    def trigger_manual_upload(self):
        """Trigger an immediate upload (doesn't wait for interval)."""
        logging.info("Manual upload triggered")
        threading.Thread(target=self._upload, daemon=True).start()

    def _upload(self):
        """Perform one upload operation."""
        try:
            logging.info("Starting upload...")

            # Get unuploaded events
            events = self.db.get_unuploaded_events(limit=self.batch_size)

            if not events:
                logging.info("No events to upload")
                self.last_upload_time = datetime.now()
                return

            logging.info(f"Found {len(events)} events to upload")

            # Upload events to cloud
            success = self._upload_events(events)

            if success:
                # Mark events as uploaded
                event_ids = [event["id"] for event in events]
                self.db.mark_events_uploaded(event_ids)

                self.total_uploaded += len(events)
                self.last_upload_time = datetime.now()
                logging.info(f"Successfully uploaded {len(events)} events")
            else:
                logging.error("Failed to upload events")
                self.upload_failures += 1

        except Exception as e:
            logging.error(f"Error during upload: {e}", exc_info=True)
            self.upload_failures += 1

    def _upload_events(self, events: list) -> bool:
        """
        Upload events to cloud platform.

        Args:
            events: List of event dictionaries from local database

        Returns:
            True if upload successful, False otherwise
        """
        if not self.cloud_config.api_key:
            logging.error("API key not configured - cannot upload events")
            return False

        try:
            # Transform events to match API schema
            api_events = []
            for event in events:
                api_event = {
                    "timestamp": event["timestamp"],
                    "speed": event["speed"],
                    "speed_limit": event["speed_limit"],
                    "is_speeding": event["is_speeding"],
                    "has_photo": bool(event.get("photo_path")),
                }
                api_events.append(api_event)

            # Prepare request
            url = f"{self.cloud_config.api_url}/api/ingest/v1/events"
            headers = {
                "X-API-Key": self.cloud_config.api_key,
                "Content-Type": "application/json",
            }
            payload = {"events": api_events}

            logging.info(
                f"Uploading {len(api_events)} events to {url}"
            )

            # Make API request with retry logic
            for attempt in range(self.retry_attempts):
                try:
                    response = requests.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=30,
                    )
                    response.raise_for_status()

                    # Parse response
                    result = response.json()
                    logging.info(
                        f"Upload successful: {result.get('processed', 0)} processed, "
                        f"{result.get('duplicates_skipped', 0)} duplicates"
                    )

                    # Handle photo uploads if any events have photos
                    if "created_events" in result:
                        self._handle_photo_uploads(events, result["created_events"])

                    return True

                except requests.exceptions.RequestException as e:
                    if attempt < self.retry_attempts - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logging.warning(
                            f"Upload attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logging.error(f"Upload failed after {self.retry_attempts} attempts: {e}")
                        return False

        except Exception as e:
            logging.error(f"Error preparing upload: {e}", exc_info=True)
            return False

        return False

    def _handle_photo_uploads(self, local_events: list, created_events: list):
        """
        Handle photo uploads for events that have photos.

        Args:
            local_events: Original event dictionaries with photo paths
            created_events: Created event info from API response with event_ids
        """
        # Create mapping of timestamp to event_id
        event_id_map = {}
        for created_event in created_events:
            if created_event.get("has_photo"):
                timestamp = created_event["timestamp"]
                event_id_map[timestamp] = created_event["event_id"]

        # Upload photos for events that have them
        for local_event in local_events:
            photo_path = local_event.get("photo_path")
            if photo_path and local_event["timestamp"] in event_id_map:
                event_id = event_id_map[local_event["timestamp"]]
                success = self._upload_photo(event_id, photo_path)
                if not success:
                    logging.error(
                        f"Failed to upload photo for event {event_id}: {photo_path}"
                    )

    def _upload_photo(self, event_id: str, photo_path: str) -> bool:
        """
        Upload photo to cloud storage using pre-signed URL workflow.

        Workflow:
        1. Request pre-signed upload URL from API
        2. Upload photo to the pre-signed URL
        3. Confirm upload completion

        Args:
            event_id: UUID of the event
            photo_path: Path to photo file

        Returns:
            True if upload successful, False otherwise
        """
        if not self.cloud_config.api_key:
            logging.error("API key not configured - cannot upload photo")
            return False

        try:
            import os
            from pathlib import Path

            photo_file = Path(photo_path)
            if not photo_file.exists():
                logging.error(f"Photo file not found: {photo_path}")
                return False

            # Step 1: Request pre-signed URL
            url_request_url = (
                f"{self.cloud_config.api_url}/api/ingest/v1/events/{event_id}/photo/url"
            )
            headers = {
                "X-API-Key": self.cloud_config.api_key,
            }

            logging.debug(f"Requesting upload URL for event {event_id}")
            response = requests.post(url_request_url, headers=headers, timeout=10)
            response.raise_for_status()

            url_data = response.json()
            upload_url = url_data["upload_url"]
            photo_key = url_data["photo_key"]

            # Handle relative URLs from API by prepending base URL
            if upload_url.startswith("/"):
                upload_url = f"{self.cloud_config.api_url}{upload_url}"

            # Step 2: Upload photo to pre-signed URL
            logging.debug(f"Uploading photo to pre-signed URL")
            with open(photo_file, "rb") as f:
                files = {'file': ('photo.jpg', f, 'image/jpeg')}
                upload_response = requests.put(upload_url, files=files, timeout=30)
            upload_response.raise_for_status()

            # Step 3: Confirm upload
            confirm_url = (
                f"{self.cloud_config.api_url}/api/ingest/v1/events/{event_id}/photo/confirm"
            )
            confirm_params = {"photo_key": photo_key}

            logging.debug(f"Confirming photo upload for event {event_id}")
            confirm_response = requests.post(
                confirm_url,
                params=confirm_params,
                headers=headers,
                timeout=10,
            )
            confirm_response.raise_for_status()

            logging.info(f"Successfully uploaded photo for event {event_id}")
            return True

        except requests.exceptions.RequestException as e:
            logging.error(f"Photo upload failed for event {event_id}: {e}")
            return False
        except Exception as e:
            logging.error(f"Error uploading photo: {e}", exc_info=True)
            return False

    def get_status(self) -> dict:
        """
        Get upload service status.

        Returns:
            Dictionary with status information
        """
        return {
            "running": self._running,
            "last_upload": self.last_upload_time.isoformat()
            if self.last_upload_time
            else None,
            "total_uploaded": self.total_uploaded,
            "upload_failures": self.upload_failures,
            "frequency_minutes": self.frequency_minutes,
        }

    def download_events(self) -> dict:
        """
        Download all events for this device from the cloud.
        Useful for restoring data after a local device reset.

        Returns:
            Dictionary with download results
        """
        if not self.cloud_config.api_key:
            logging.error("API key not configured - cannot download events")
            return {
                "success": False,
                "message": "API key not configured",
                "downloaded": 0,
            }

        try:
            # Request events from cloud API
            url = f"{self.cloud_config.api_url}/api/ingest/v1/events"
            headers = {
                "X-API-Key": self.cloud_config.api_key,
            }

            logging.info(f"Downloading events from {url}")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            # Parse response
            data = response.json()
            events = data.get("events", [])

            if not events:
                logging.info("No events available to download from cloud")
                return {
                    "success": True,
                    "message": "No events available on cloud",
                    "downloaded": 0,
                }

            logging.info(f"Downloaded {len(events)} events from cloud")

            # Insert events into local database
            # Note: We're not marking these as uploaded since they came from the cloud
            inserted_count = 0
            for event in events:
                try:
                    # Insert event into database
                    # The database should handle duplicates appropriately
                    self.db.add_event(
                        timestamp=event.get("timestamp"),
                        speed=event.get("speed"),
                        speed_limit=event.get("speed_limit", self.db._config.device.speed_limit),
                        is_speeding=event.get("is_speeding", False),
                        photo_path=None,  # Photos are not downloaded
                        uploaded=True,  # Mark as uploaded since it came from cloud
                    )
                    inserted_count += 1
                except Exception as e:
                    logging.debug(f"Skipping event (possibly duplicate): {e}")
                    continue

            logging.info(f"Successfully inserted {inserted_count} events into local database")
            return {
                "success": True,
                "message": f"Downloaded {len(events)} events, inserted {inserted_count} new events",
                "downloaded": len(events),
                "inserted": inserted_count,
            }

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download events: {e}")
            return {
                "success": False,
                "message": f"Download failed: {str(e)}",
                "downloaded": 0,
            }
        except Exception as e:
            logging.error(f"Error downloading events: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "downloaded": 0,
            }
