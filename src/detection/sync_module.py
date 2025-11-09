"""
Synchronized detection module that integrates radar speed detection with camera capture.
Implements state machine for vehicle tracking and event recording.
"""

import time
import logging
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass

from ..sensors import RadarSensor, Camera, ObjectDetector
from ..sensors.smart_capture import SmartCaptureModule
from ..database import SQLiteManager


@dataclass
class DetectionEvent:
    """Data class for a speed detection event."""

    timestamp: datetime
    speed: float
    speed_limit: float
    is_speeding: bool
    photo_paths: list[str]


class SyncDetectionModule:
    """Manages synchronized radar detection and camera capture."""

    def __init__(
        self,
        radar: RadarSensor,
        camera: Camera,
        db: SQLiteManager,
        object_detector: Optional[ObjectDetector] = None,
        speed_limit: float = 25.0,
        photo_trigger_threshold: float = 5.0,  # mph over limit
        min_track_time: float = 0.1,  # seconds
        targetless_timeout: float = 0.2,  # seconds
        idle_notice_interval: float = 10.0,  # seconds
        smart_capture_window: float = 2.0,  # seconds to find best shot
        continuous_detection: bool = True,  # run detection during tracking
        detection_interval: float = 0.3,  # seconds between detection checks
    ):
        """
        Initialize synchronized detection module.

        Args:
            radar: Radar sensor instance
            camera: Camera instance
            db: Database manager instance
            object_detector: Optional object detector for smart capture
            speed_limit: Speed limit for the road (MPH)
            photo_trigger_threshold: How many MPH over limit to trigger photo
            min_track_time: Minimum time to track before target acquired
            targetless_timeout: Time without valid reading before giving up
            idle_notice_interval: Time between idle notifications
            smart_capture_window: Time window to find best-centered shot
            continuous_detection: Enable detection logging during all tracking
            detection_interval: Seconds between detection checks during tracking
        """
        self.radar = radar
        self.camera = camera
        self.db = db
        self.object_detector = object_detector
        self.speed_limit = speed_limit
        self.photo_trigger_threshold = photo_trigger_threshold
        self.min_track_time = min_track_time
        self.targetless_timeout = targetless_timeout
        self.idle_notice_interval = idle_notice_interval
        self.continuous_detection = continuous_detection
        self.detection_interval = detection_interval

        # Initialize smart capture if object detector available
        self.smart_capture = None
        if object_detector and object_detector._is_initialized:
            self.smart_capture = SmartCaptureModule(
                camera=camera,
                object_detector=object_detector,
                capture_window_seconds=smart_capture_window
            )
            logging.info("Smart capture enabled - will use object detection for optimal photo timing")

        # Tracking state
        self.tracking = False
        self.target_acquired = False
        self.recent_velocity: Optional[float] = None
        self.prior_velocity: Optional[float] = None
        self.max_speed_seen: float = 0.0
        self.tracking_start_time: float = 0.0
        self.targetless_start_time: Optional[float] = None

        # Statistics
        self.total_events = 0
        self.total_speeders = 0

        # Event callbacks (can be overridden)
        self.on_target_acquired_callback: Optional[Callable] = None
        self.on_target_lost_callback: Optional[Callable] = None
        self.on_idle_callback: Optional[Callable] = None

    def run(self, stop_event: Optional[any] = None):
        """
        Run the main detection loop.

        Args:
            stop_event: Threading event to signal stop (optional)
        """
        logging.info("Starting synchronized detection module")

        try:
            while True:
                # Check stop signal
                if stop_event and stop_event.is_set():
                    break

                # Run one cycle of the state machine
                self._detection_cycle()

        except KeyboardInterrupt:
            logging.info("Detection loop interrupted")
        except Exception as e:
            logging.error(f"Error in detection loop: {e}", exc_info=True)
        finally:
            logging.info("Detection module stopped")

    def _detection_cycle(self):
        """Run one cycle of the detection state machine."""
        # Check if radar is still initialized
        if not self.radar._is_initialized or not self.radar.serial_port:
            logging.debug("Radar sensor not available, skipping detection cycle")
            time.sleep(0.1)
            return

        # Flush radar buffers at start of cycle
        if hasattr(self.radar.serial_port, 'flushInput'):
            try:
                self.radar.serial_port.flushInput()
                self.radar.serial_port.flushOutput()
            except Exception:
                # Sensor was closed
                return

        # Reset tracking state
        self.tracking = False
        self.target_acquired = False
        self.max_speed_seen = 0.0

        # Phase 1: Wait for valid speed (not tracking)
        self._wait_for_target()

        # Phase 2: Track target and collect speed data
        self._track_target()

    def _wait_for_target(self):
        """Wait for a valid speed reading to start tracking."""
        idle_start_time = time.time()

        while not self.tracking:
            velocity = self.radar.read_velocity()

            if velocity is not None:
                self.recent_velocity = velocity
                is_valid = self.radar.is_speed_valid(velocity)

                if is_valid:
                    # Valid speed detected, start tracking
                    self.tracking = True
                    self.tracking_start_time = time.time()
                    logging.info(f"Target detected: {abs(velocity)} MPH")
                    break

                # Check for idle timeout
                if self.idle_notice_interval > 0:
                    elapsed = time.time() - idle_start_time
                    if elapsed > self.idle_notice_interval:
                        self._on_idle()
                        idle_start_time = time.time()

    def _track_target(self):
        """Track target and record data until target is lost."""
        self.targetless_start_time = None
        last_detection_time = 0  # Track last object detection check

        while self.tracking:
            self.prior_velocity = self.recent_velocity
            velocity = self.radar.read_velocity()
            current_time = time.time()

            if velocity is None:
                # No reading from radar - check if we've been targetless too long
                if self.targetless_start_time is None:
                    self.targetless_start_time = current_time
                else:
                    elapsed = current_time - self.targetless_start_time
                    if elapsed > self.targetless_timeout:
                        # Lost target due to no readings
                        if self.target_acquired:
                            self._on_target_lost(capture_photo=True)
                        elif self.max_speed_seen > 0:
                            # Store event even if target wasn't formally acquired
                            # This handles fast-moving vehicles that pass quickly
                            logging.info(f"Quick detection, storing event without photo (max speed: {abs(self.max_speed_seen)} MPH)")
                            self._on_target_lost(capture_photo=False)
                        else:
                            logging.debug("Target lost before any valid speed recorded")
                        logging.debug("Target lost: no radar readings")
                        self.tracking = False
                        self.target_acquired = False
                continue

            self.recent_velocity = velocity

            # Run object detection periodically during tracking if enabled
            if (self.continuous_detection and
                self.object_detector and
                self.object_detector._is_initialized):
                if current_time - last_detection_time >= self.detection_interval:
                    try:
                        frame = self.camera.get_current_frame()
                        if frame is not None:
                            detected, bbox, metadata = self.object_detector.detect_with_metadata(frame)
                            if detected and metadata:
                                logging.info(
                                    f"Object detected during tracking: {metadata['class_name']} "
                                    f"(confidence: {metadata['confidence']:.2f}, "
                                    f"speed: {abs(velocity):.0f} MPH)"
                                )
                    except Exception as e:
                        logging.debug(f"Detection check failed: {e}")
                    last_detection_time = current_time

            # Check if speed is in valid range
            if self.radar.is_speed_valid(velocity):
                # Reset targetless timer
                self.targetless_start_time = None

                # Track maximum speed
                if abs(velocity) > abs(self.max_speed_seen):
                    self.max_speed_seen = abs(velocity)

                # Check direction consistency
                same_direction = self._is_same_direction(
                    self.prior_velocity, velocity
                )

                if same_direction:
                    # Consistent tracking
                    track_duration = current_time - self.tracking_start_time

                    if track_duration > self.min_track_time:
                        if not self.target_acquired:
                            # Target acquired for first time
                            self._on_target_acquired(velocity)
                            self.target_acquired = True
                        else:
                            # Continue tracking acquired target
                            if abs(velocity) > abs(self.prior_velocity or 0):
                                logging.debug(f"Acceleration: {abs(velocity)} MPH")

                else:
                    # Direction changed
                    if self.target_acquired:
                        # Lost current target, may be new target
                        logging.info("Direction change, target lost")
                        self._on_target_lost()

                    # Reset tracking for new direction
                    self.target_acquired = False
                    self.prior_velocity = 0
                    self.max_speed_seen = abs(velocity)
                    self.tracking_start_time = current_time
                    self.targetless_start_time = None

            else:
                # Speed out of valid range
                if self.targetless_start_time is None:
                    self.targetless_start_time = current_time
                else:
                    elapsed = current_time - self.targetless_start_time

                    if elapsed > self.targetless_timeout:
                        # Target lost
                        if self.target_acquired:
                            self._on_target_lost()

                        self.tracking = False
                        self.target_acquired = False

    def _is_same_direction(
        self, prev_velocity: Optional[float], curr_velocity: float
    ) -> bool:
        """Check if two velocities are in the same direction."""
        if prev_velocity is None:
            return True

        # Same direction if both positive or both negative
        return (prev_velocity > 0 and curr_velocity > 0) or (
            prev_velocity < 0 and curr_velocity < 0
        )

    def _on_target_acquired(self, velocity: float):
        """Handle target acquired event."""
        direction = "inbound" if velocity > 0 else "outbound"
        logging.info(f"Target acquired: {abs(velocity)} MPH ({direction})")

        if self.on_target_acquired_callback:
            self.on_target_acquired_callback(velocity)

    def _on_target_lost(self, capture_photo: bool = True):
        """
        Handle target lost event - record event and capture photo if needed.

        Args:
            capture_photo: Whether to attempt photo capture for speeders
        """
        speed = abs(self.max_speed_seen)
        is_speeding = speed > self.speed_limit

        logging.info(
            f"Target lost: max speed {speed} MPH "
            f"({'SPEEDING' if is_speeding else 'within limit'})"
        )

        # Decide if we should take a photo
        should_photo = capture_photo and speed >= (self.speed_limit + self.photo_trigger_threshold)
        photo_path = None

        if should_photo:
            logging.info(f"Triggering smart capture for speeder: {speed} MPH")

            # Use smart capture if available, otherwise fallback to immediate capture
            if self.smart_capture:
                photo_path = self.smart_capture.start_smart_capture()
            else:
                # Fallback to old method - immediate capture
                timestamp = time.time()
                photo_path = self.camera.capture_photo(timestamp)

        # Store event in database
        event_id = self.db.store_event(
            speed=speed,
            speed_limit=self.speed_limit,
            is_speeding=is_speeding,
            photo_path=photo_path,
        )

        # Update statistics
        self.total_events += 1
        if is_speeding:
            self.total_speeders += 1

        logging.info(
            f"Event recorded: ID={event_id}, speed={speed}, "
            f"photo={'captured' if photo_path else 'none'}"
        )

        # Callback
        if self.on_target_lost_callback:
            self.on_target_lost_callback(
                DetectionEvent(
                    timestamp=datetime.now(),
                    speed=speed,
                    speed_limit=self.speed_limit,
                    is_speeding=is_speeding,
                    photo_paths=[photo_path] if photo_path else [],
                )
            )

    def _on_idle(self):
        """Handle idle notification."""
        logging.debug("System idle, waiting for vehicles...")

        if self.on_idle_callback:
            self.on_idle_callback()

    def get_statistics(self) -> dict:
        """
        Get current statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_events": self.total_events,
            "total_speeders": self.total_speeders,
            "tracking": self.tracking,
            "target_acquired": self.target_acquired,
            "recent_speed": abs(self.recent_velocity) if self.recent_velocity else 0,
        }
