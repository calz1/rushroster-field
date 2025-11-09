"""
Burst capture module that rapidly captures frames when a speeder is detected.
Frames are saved to temporary storage and post-processed asynchronously with YOLO.
"""

import logging
import time
import cv2
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BurstCaptureResult:
    """Result of a burst capture session."""
    event_id: int
    frame_paths: List[str]
    capture_count: int
    duration: float
    temp_dir: str


class BurstCaptureModule:
    """
    Rapidly captures frames during a speeding event for post-processing.
    Optimized for minimal CPU usage during capture - no object detection.
    """

    def __init__(
        self,
        camera: any,
        capture_interval: float = 0.1,  # 100ms between captures
        max_frames: int = 20,  # Cap at 20 frames (2 seconds)
        temp_dir: str = "photos/temp_burst",
    ):
        """
        Initialize burst capture module.

        Args:
            camera: Camera instance
            capture_interval: Seconds between frame captures (0.1 = 100ms)
            max_frames: Maximum frames to capture per event
            temp_dir: Temporary directory for burst frames
        """
        self.camera = camera
        self.capture_interval = capture_interval
        self.max_frames = max_frames
        self.temp_dir = Path(temp_dir)

        # Create temp directory
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        logging.info(
            f"Burst capture initialized: interval={capture_interval*1000:.0f}ms, "
            f"max_frames={max_frames}, temp_dir={temp_dir}"
        )

    def start_burst_capture(self, event_id: int) -> Optional[BurstCaptureResult]:
        """
        Start burst capture for a speeding event.
        Captures frames rapidly without object detection for later processing.

        Args:
            event_id: Database event ID for this speeding event

        Returns:
            BurstCaptureResult with paths to captured frames, or None if failed
        """
        logging.info(f"Starting burst capture for event {event_id}")

        start_time = time.time()
        frame_paths = []
        capture_count = 0

        # Create event-specific temp directory
        event_temp_dir = self.temp_dir / f"event_{event_id}"
        event_temp_dir.mkdir(exist_ok=True)

        try:
            while capture_count < self.max_frames:
                # Get current frame from camera
                frame = self.camera.get_current_frame()

                if frame is None:
                    logging.warning("Failed to get frame from camera")
                    time.sleep(self.capture_interval)
                    continue

                # Save frame to temp directory
                frame_filename = f"frame_{capture_count:03d}.jpg"
                frame_path = event_temp_dir / frame_filename

                success = cv2.imwrite(str(frame_path), frame)

                if success:
                    frame_paths.append(str(frame_path))
                    capture_count += 1
                    logging.debug(f"Captured frame {capture_count}/{self.max_frames}: {frame_path}")
                else:
                    logging.warning(f"Failed to save frame {frame_path}")

                # Wait before next capture
                time.sleep(self.capture_interval)

            duration = time.time() - start_time

            logging.info(
                f"Burst capture complete for event {event_id}: "
                f"captured {capture_count} frames in {duration:.2f}s"
            )

            return BurstCaptureResult(
                event_id=event_id,
                frame_paths=frame_paths,
                capture_count=capture_count,
                duration=duration,
                temp_dir=str(event_temp_dir)
            )

        except Exception as e:
            logging.error(f"Error during burst capture for event {event_id}: {e}")
            return None

    def start_burst_capture_timed(
        self,
        event_id: int,
        duration_seconds: float = 2.0
    ) -> Optional[BurstCaptureResult]:
        """
        Start burst capture for a specific duration (alternative to frame count).

        Args:
            event_id: Database event ID for this speeding event
            duration_seconds: How long to capture frames

        Returns:
            BurstCaptureResult with paths to captured frames, or None if failed
        """
        logging.info(f"Starting timed burst capture for event {event_id} ({duration_seconds}s)")

        start_time = time.time()
        frame_paths = []
        capture_count = 0

        # Create event-specific temp directory
        event_temp_dir = self.temp_dir / f"event_{event_id}"
        event_temp_dir.mkdir(exist_ok=True)

        try:
            while (time.time() - start_time) < duration_seconds:
                # Get current frame from camera
                frame = self.camera.get_current_frame()

                if frame is None:
                    logging.warning("Failed to get frame from camera")
                    time.sleep(self.capture_interval)
                    continue

                # Save frame to temp directory
                frame_filename = f"frame_{capture_count:03d}.jpg"
                frame_path = event_temp_dir / frame_filename

                success = cv2.imwrite(str(frame_path), frame)

                if success:
                    frame_paths.append(str(frame_path))
                    capture_count += 1
                    logging.debug(f"Captured frame {capture_count}: {frame_path}")
                else:
                    logging.warning(f"Failed to save frame {frame_path}")

                # Wait before next capture
                time.sleep(self.capture_interval)

            duration = time.time() - start_time

            logging.info(
                f"Timed burst capture complete for event {event_id}: "
                f"captured {capture_count} frames in {duration:.2f}s"
            )

            return BurstCaptureResult(
                event_id=event_id,
                frame_paths=frame_paths,
                capture_count=capture_count,
                duration=duration,
                temp_dir=str(event_temp_dir)
            )

        except Exception as e:
            logging.error(f"Error during timed burst capture for event {event_id}: {e}")
            return None

    def cleanup_event_frames(self, event_id: int):
        """
        Clean up all temporary frames for an event.

        Args:
            event_id: Event ID to clean up
        """
        event_temp_dir = self.temp_dir / f"event_{event_id}"

        if event_temp_dir.exists():
            try:
                import shutil
                shutil.rmtree(event_temp_dir)
                logging.debug(f"Cleaned up temp frames for event {event_id}")
            except Exception as e:
                logging.warning(f"Failed to cleanup temp frames for event {event_id}: {e}")
