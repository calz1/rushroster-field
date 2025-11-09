"""
Smart capture module that uses object detection to determine best photo timing.
Waits for vehicle to be optimally centered in frame before capturing.
"""

import logging
import time
import threading
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class CaptureCandidate:
    """A candidate photo with its centering score."""
    frame: any  # numpy array
    bbox: Tuple[int, int, int, int]
    timestamp: float
    centering_score: float  # 0-1, higher is better


class SmartCaptureModule:
    """
    Monitors camera feed with object detection to capture vehicle at optimal moment.
    Captures when vehicle is best centered in frame.
    """

    def __init__(
        self,
        camera: any,
        object_detector: any,
        capture_window_seconds: float = 2.0,
        detection_interval: float = 0.1,  # Run detection every 100ms
    ):
        """
        Initialize smart capture module.

        Args:
            camera: Camera instance
            object_detector: ObjectDetector instance
            capture_window_seconds: How long to watch for best shot after trigger
            detection_interval: How often to run object detection (seconds)
        """
        self.camera = camera
        self.object_detector = object_detector
        self.capture_window_seconds = capture_window_seconds
        self.detection_interval = detection_interval

        self._active = False
        self._best_candidate: Optional[CaptureCandidate] = None
        self._capture_start_time: Optional[float] = None

    def start_smart_capture(self) -> Optional[str]:
        """
        Start monitoring for the best-centered vehicle shot.
        Runs for capture_window_seconds and returns the best frame.

        Returns:
            Path to saved photo, or None if no good shot found
        """
        if not self.object_detector._is_initialized:
            logging.warning("Object detector not initialized, falling back to immediate capture")
            return self._fallback_capture()

        self._active = True
        self._best_candidate = None
        self._capture_start_time = time.time()

        logging.info("Starting smart capture - looking for best-centered vehicle...")

        # Monitor frames for capture window duration
        while self._active:
            elapsed = time.time() - self._capture_start_time

            # Check if capture window expired
            if elapsed >= self.capture_window_seconds:
                break

            # Get current frame
            frame = self.camera.get_current_frame()
            if frame is None:
                time.sleep(self.detection_interval)
                continue

            # Run object detection
            vehicle_detected, bbox = self.object_detector.detect_vehicles(frame)

            if vehicle_detected and bbox:
                # Calculate centering score
                centering_score = self._calculate_centering_score(bbox, frame.shape)

                # Check if this is the best candidate so far
                if self._best_candidate is None or centering_score > self._best_candidate.centering_score:
                    self._best_candidate = CaptureCandidate(
                        frame=frame.copy(),
                        bbox=bbox,
                        timestamp=time.time(),
                        centering_score=centering_score
                    )

                    logging.debug(
                        f"New best candidate: centering_score={centering_score:.2f}, "
                        f"bbox_center={self._get_bbox_center(bbox)}"
                    )

                    # If we got a really good shot (>90% centered), capture immediately
                    if centering_score > 0.9:
                        logging.info(f"Excellent shot found (score: {centering_score:.2f}), capturing now")
                        break

            # Wait before next detection
            time.sleep(self.detection_interval)

        # Save the best candidate if we found one
        if self._best_candidate:
            photo_path = self._save_best_frame()
            logging.info(
                f"Smart capture complete: saved best frame with centering score "
                f"{self._best_candidate.centering_score:.2f}"
            )
            return photo_path
        else:
            logging.warning("No vehicle detected during capture window, using fallback")
            return self._fallback_capture()

    def _calculate_centering_score(
        self,
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int, int]
    ) -> float:
        """
        Calculate how well-centered the vehicle is in the frame.

        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            frame_shape: Frame shape (height, width, channels)

        Returns:
            Score from 0-1, where 1.0 is perfectly centered
        """
        height, width = frame_shape[:2]

        # Get bbox center
        bbox_center_x, bbox_center_y = self._get_bbox_center(bbox)

        # Get frame center
        frame_center_x = width / 2
        frame_center_y = height / 2

        # Calculate distance from center (normalized)
        x_distance = abs(bbox_center_x - frame_center_x) / (width / 2)
        y_distance = abs(bbox_center_y - frame_center_y) / (height / 2)

        # Combined distance (0 = perfect center, 1 = edge)
        distance = (x_distance + y_distance) / 2

        # Convert to score (1 = perfect center, 0 = edge)
        score = 1.0 - distance

        # Also consider bbox size (larger is better, vehicle is closer)
        bbox_area = self.object_detector.get_bbox_size(bbox)
        frame_area = width * height
        size_ratio = bbox_area / frame_area

        # Combine centering and size (70% centering, 30% size)
        # Size should be between 5-40% of frame for good shot
        optimal_size_ratio = 0.20  # 20% is ideal
        size_score = 1.0 - min(abs(size_ratio - optimal_size_ratio) / optimal_size_ratio, 1.0)

        final_score = (score * 0.7) + (size_score * 0.3)

        return final_score

    def _get_bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y

    def _save_best_frame(self) -> Optional[str]:
        """Save the best candidate frame to disk."""
        if not self._best_candidate:
            return None

        import cv2
        import os
        from datetime import datetime

        # Create filename
        timestamp = self._best_candidate.timestamp
        dt = datetime.fromtimestamp(timestamp)
        filename = dt.strftime("car_%Y_%m_%d_%H-%M-%S.jpg")
        filepath = os.path.join(self.camera.output_dir, filename)

        # Save frame
        cv2.imwrite(filepath, self._best_candidate.frame)

        # Optionally save annotated version for debugging
        if logging.getLogger().level == logging.DEBUG:
            annotated = self.object_detector.annotate_frame(
                self._best_candidate.frame,
                self._best_candidate.bbox
            )
            debug_filepath = filepath.replace(".jpg", "_annotated.jpg")
            cv2.imwrite(debug_filepath, annotated)
            logging.debug(f"Saved annotated frame: {debug_filepath}")

        logging.info(f"Saved best frame: {filepath}")
        return filepath

    def _fallback_capture(self) -> Optional[str]:
        """Fallback to immediate capture if object detection unavailable."""
        timestamp = time.time()
        return self.camera.capture_photo(timestamp)

    def stop(self):
        """Stop active capture."""
        self._active = False
