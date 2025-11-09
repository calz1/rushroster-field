"""Upload, download, and synchronization endpoints."""

import logging
import threading
from typing import Any

from fastapi import FastAPI, HTTPException


def register_sync_routes(
    app: FastAPI,
    db: Any,
    sync_service: Any,
) -> None:
    """
    Register synchronization endpoints.

    Args:
        app: FastAPI application instance
        db: Database manager instance
        sync_service: Sync service instance for triggering manual syncs (optional)
    """

    @app.post("/upload")
    async def trigger_upload():
        """
        Trigger manual upload to cloud.
        """
        try:
            if sync_service:
                # Trigger immediate upload
                sync_service.trigger_manual_upload()

                # Get queue size to report to user
                unuploaded = db.get_unuploaded_events()
                queue_size = len(unuploaded)

                logging.info(f"Manual upload triggered - {queue_size} events in queue")
                return {
                    "status": "ok",
                    "message": f"Upload triggered - uploading {queue_size} event(s)",
                    "queue_size": queue_size,
                }
            else:
                return {
                    "status": "error",
                    "message": "Upload service not available",
                }

        except Exception as e:
            logging.error(f"Error triggering upload: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/download")
    async def trigger_download():
        """
        Trigger download of events from cloud to local device.
        Useful for restoring data after device reset.
        """
        try:
            if sync_service:
                # Trigger download in background thread
                def download_task():
                    result = sync_service.download_events()
                    logging.info(f"Download completed: {result}")

                threading.Thread(target=download_task, daemon=True).start()

                logging.info("Download initiated")
                return {
                    "status": "ok",
                    "message": "Download initiated - fetching events from cloud",
                }
            else:
                return {
                    "status": "error",
                    "message": "Upload service not available",
                }

        except Exception as e:
            logging.error(f"Error triggering download: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/force-resync")
    async def force_resync():
        """
        Reset all events to not uploaded status, forcing a complete re-sync.
        Useful after photo upload errors or cloud sync issues.
        """
        try:
            # Get count of events that will be reset
            total_events = db.get_total_event_count()

            # Mark all events as not uploaded
            db.reset_all_upload_status()

            logging.info(f"Force resync triggered - marked {total_events} event(s) for re-upload")
            return {
                "status": "ok",
                "message": f"Force resync complete - {total_events} event(s) marked for re-upload",
                "events_reset": total_events,
            }

        except Exception as e:
            logging.error(f"Error during force resync: {e}")
            raise HTTPException(status_code=500, detail=str(e))
