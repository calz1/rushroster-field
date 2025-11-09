"""
Burst detection module - optimized for Raspberry Pi.
Uses fast burst capture during speeding events, then post-processes with YOLO asynchronously.
"""

import time
import logging
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass

from ..sensors import RadarSensor, Camera, ObjectDetector
from ..sensors.burst_capture import BurstCaptureModule
from ..sensors.async_frame_processor import AsyncFrameProcessor, ProcessingResult
from ..database import SQLiteManager


@dataclass
class DetectionEvent:
    """Data class for a speed detection event."""

    timestamp: datetime
    speed: float
    speed_limit: float
    is_speeding: bool
    photo_paths: list[str]


class BurstDetectionModule:
    """
    Manages synchronized radar detection with burst camera capture.
    Optimized for Raspberry Pi - no real-time YOLO, only post-processing.
    """

    def __init__(
        self,
        radar: RadarSensor,
        camera: Camera,
        db: SQLiteManager,
        object_detector_config: Optional[dict] = None,
        speed_limit: float = 25.0,
        photo_trigger_threshold: float = 5.0,  # mph over limit
        min_track_time: float = 0.1,  # seconds
        targetless_timeout: float = 0.2,  # seconds
        idle_notice_interval: float = 10.0,  # seconds
        burst_capture_interval: float = 0.1,  # 100ms between frames
        burst_max_frames: int = 20,  # Max frames to capture
        event_cooldown: float = 1.0,  # seconds to group detections as same event
    ):
        """
        Initialize burst detection module.

        Args:
            radar: Radar sensor instance
            camera: Camera instance
            db: Database manager instance
            object_detector_config: Config dict for object detector (optional)
            speed_limit: Speed limit for the road (MPH)
            photo_trigger_threshold: How many MPH over limit to trigger photo
            min_track_time: Minimum time to track before target acquired
            targetless_timeout: Time without valid reading before giving up
            idle_notice_interval: Time between idle notifications
            burst_capture_interval: Seconds between burst captures (0.1 = 100ms)
            burst_max_frames: Maximum frames to capture per event
            event_cooldown: Seconds to group detections as same event
        """
        self.radar = radar
        self.camera = camera
        self.db = db
        self.speed_limit = speed_limit
        self.photo_trigger_threshold = photo_trigger_threshold
        self.min_track_time = min_track_time
        self.targetless_timeout = targetless_timeout
        self.idle_notice_interval = idle_notice_interval
        self.event_cooldown = event_cooldown

        # Initialize burst capture module
        self.burst_capture = BurstCaptureModule(
            camera=camera,
            capture_interval=burst_capture_interval,
            max_frames=burst_max_frames,
        )

        # Initialize async frame processor
        if object_detector_config:
            self.frame_processor = AsyncFrameProcessor(
                output_dir="photos",
                model_dir=object_detector_config.get("model_dir", "darknet_models"),
                model_size=object_detector_config.get("model_size", "tiny"),
                confidence_threshold=object_detector_config.get("confidence_threshold", 0.5),
                detection_mode=object_detector_config.get("detection_mode", "all_moving"),
                inference_size=object_detector_config.get("inference_size", 320),
                nms_threshold=object_detector_config.get("nms_threshold", 0.4),
            )
            logging.info("Burst detection with async YOLO post-processing enabled")
        else:
            self.frame_processor = None
            logging.warning("No object detector config provided - burst capture only")

        # Tracking state
        self.tracking = False
        self.target_acquired = False
        self.recent_velocity: Optional[float] = None
        self.prior_velocity: Optional[float] = None
        self.max_speed_seen: float = 0.0
        self.tracking_start_time: float = 0.0
        self.targetless_start_time: Optional[float] = None

        # Event grouping state (to prevent multiple events for same vehicle)
        self.last_event_time: Optional[float] = None
        self.cooldown_max_speed: float = 0.0
        self.cooldown_had_burst: bool = False

        # Burst capture state
        self.capturing_burst = False
        self.current_event_id: Optional[int] = None

        # Statistics
        self.total_events = 0
        self.total_speeders = 0

        # Event callbacks
        self.on_target_acquired_callback: Optional[Callable] = None
        self.on_target_lost_callback: Optional[Callable] = None
        self.on_idle_callback: Optional[Callable] = None

    def run(self, stop_event: Optional[any] = None):
        """
        Run the main detection loop.

        Args:
            stop_event: Threading event to signal stop (optional)
        """
        logging.info("Starting burst detection module (optimized for Raspberry Pi)")

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
            logging.info("Burst detection module stopped")

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

        # Check if we're in cooldown period (grouping detections)
        current_time = time.time()
        in_cooldown = False

        if self.last_event_time is not None:
            time_since_last_event = current_time - self.last_event_time
            in_cooldown = time_since_last_event < self.event_cooldown

            if not in_cooldown:
                # Cooldown expired - finalize the grouped event
                self._finalize_cooldown_event()

        # Reset tracking state (but preserve cooldown state)
        self.tracking = False
        self.target_acquired = False
        if not in_cooldown:
            self.max_speed_seen = 0.0
            self.capturing_burst = False
            self.current_event_id = None

        # Phase 1: Wait for valid speed (not tracking)
        self._wait_for_target()

        # Phase 2: Track target and collect speed data
        self._track_target()

    def _wait_for_target(self):
        """Wait for a valid speed reading to start tracking."""
        idle_start_time = time.time()

        while not self.tracking:
            # Check if cooldown has expired and needs finalization
            if self.last_event_time is not None:
                time_since_last_event = time.time() - self.last_event_time
                if time_since_last_event >= self.event_cooldown:
                    # Cooldown expired - finalize the grouped event
                    self._finalize_cooldown_event()

            velocity = self.radar.read_velocity()

            if velocity is not None:
                self.recent_velocity = velocity
                is_valid = self.radar.is_speed_valid(velocity)

                if is_valid:
                    # Valid speed detected, start tracking
                    self.tracking = True
                    self.tracking_start_time = time.time()

                    # Check if we're in cooldown (grouping with previous detection)
                    if self.last_event_time is not None:
                        time_since_last = time.time() - self.last_event_time
                        if time_since_last < self.event_cooldown:
                            logging.info(f"Target detected: {abs(velocity)} MPH (grouping with previous event, {time_since_last:.1f}s ago)")
                        else:
                            logging.info(f"Target detected: {abs(velocity)} MPH")
                    else:
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
                            self._on_target_lost()
                        elif self.max_speed_seen > 0:
                            # Store event even if target wasn't formally acquired
                            logging.info(f"Quick detection, storing event (max speed: {abs(self.max_speed_seen)} MPH)")
                            self._on_target_lost()
                        else:
                            logging.debug("Target lost before any valid speed recorded")
                        logging.debug("Target lost: no radar readings")
                        self.tracking = False
                        self.target_acquired = False
                continue

            self.recent_velocity = velocity

            # Check if speed is in valid range
            if self.radar.is_speed_valid(velocity):
                # Reset targetless timer
                self.targetless_start_time = None

                # Track maximum speed
                if abs(velocity) > abs(self.max_speed_seen):
                    self.max_speed_seen = abs(velocity)

                    # Also update cooldown max speed if we're tracking higher speeds
                    if abs(velocity) > self.cooldown_max_speed:
                        self.cooldown_max_speed = abs(velocity)

                    # Check if we should start burst capture
                    if not self.capturing_burst:
                        if abs(velocity) >= (self.speed_limit + self.photo_trigger_threshold):
                            self._start_burst_capture()

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
                    self.capturing_burst = False

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

    def _start_burst_capture(self):
        """Start burst capture for a speeding vehicle."""
        self.capturing_burst = True

        # Create a temporary event ID for burst capture
        # We'll update it later with the photo path
        event_id = self.db.store_event(
            speed=abs(self.max_speed_seen),
            speed_limit=self.speed_limit,
            is_speeding=True,
            photo_path=None,  # Will be updated after processing
        )

        self.current_event_id = event_id

        logging.info(f"Starting burst capture for event {event_id} (speed: {abs(self.max_speed_seen)} MPH)")

        # Start burst capture in background (non-blocking)
        import threading

        def capture_burst():
            burst_result = self.burst_capture.start_burst_capture(event_id)

            if burst_result and self.frame_processor:
                # Start async processing
                self.frame_processor.process_burst_async(
                    event_id=event_id,
                    frame_paths=burst_result.frame_paths,
                    temp_dir=burst_result.temp_dir,
                    callback=self._on_processing_complete
                )

        capture_thread = threading.Thread(target=capture_burst, daemon=True)
        capture_thread.start()

    def _on_processing_complete(self, result: ProcessingResult):
        """
        Callback when async frame processing completes.

        Args:
            result: Processing result with best frame path
        """
        logging.info(
            f"Processing complete for event {result.event_id}: "
            f"{result.frames_with_detection}/{result.total_frames} frames had detections, "
            f"best_score={result.best_score:.2f}, "
            f"processing_time={result.processing_time:.1f}s"
        )

        # Update database with final photo path
        if result.best_frame_path:
            self.db.update_event_photo(result.event_id, result.best_frame_path)
            logging.info(f"Updated event {result.event_id} with best frame: {result.best_frame_path}")

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

    def _on_target_lost(self):
        """Handle target lost event - start cooldown period to group detections."""
        speed = abs(self.max_speed_seen)
        is_speeding = speed > self.speed_limit

        logging.info(
            f"Target lost: max speed {speed} MPH "
            f"({'SPEEDING' if is_speeding else 'within limit'})"
        )

        # Start or update cooldown period
        current_time = time.time()

        # Update cooldown max speed if this is higher
        if speed > self.cooldown_max_speed:
            self.cooldown_max_speed = speed

        # Track if we had a burst capture during this cooldown
        if self.capturing_burst:
            self.cooldown_had_burst = True

        # Update last event time to extend cooldown
        self.last_event_time = current_time

    def _finalize_cooldown_event(self):
        """Finalize the grouped event after cooldown expires."""
        if self.cooldown_max_speed == 0.0:
            # No valid speeds during cooldown
            self._reset_cooldown_state()
            return

        speed = self.cooldown_max_speed
        is_speeding = speed > self.speed_limit

        logging.info(
            f"Event finalized: max speed {speed} MPH "
            f"({'SPEEDING' if is_speeding else 'within limit'})"
        )

        # Only store event if we haven't already (burst capture creates event)
        if not self.cooldown_had_burst:
            event_id = self.db.store_event(
                speed=speed,
                speed_limit=self.speed_limit,
                is_speeding=is_speeding,
                photo_path=None,
            )

        # Update statistics
        self.total_events += 1
        if is_speeding:
            self.total_speeders += 1

        # Callback
        if self.on_target_lost_callback:
            self.on_target_lost_callback(
                DetectionEvent(
                    timestamp=datetime.now(),
                    speed=speed,
                    speed_limit=self.speed_limit,
                    is_speeding=is_speeding,
                    photo_paths=[],
                )
            )

        # Reset cooldown state
        self._reset_cooldown_state()

    def _reset_cooldown_state(self):
        """Reset the cooldown tracking state."""
        self.last_event_time = None
        self.cooldown_max_speed = 0.0
        self.cooldown_had_burst = False

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
        active_processing = 0
        if self.frame_processor:
            active_processing = self.frame_processor.get_active_count()

        return {
            "total_events": self.total_events,
            "total_speeders": self.total_speeders,
            "tracking": self.tracking,
            "target_acquired": self.target_acquired,
            "recent_speed": abs(self.recent_velocity) if self.recent_velocity else 0,
            "capturing_burst": self.capturing_burst,
            "active_processing_jobs": active_processing,
        }
