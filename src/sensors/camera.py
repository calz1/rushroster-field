"""
Camera interface for synchronized photo capture with radar detections.
Based on working implementation from original RushRoster project.
"""

import os
import logging
import threading
import time
from datetime import datetime
from typing import Optional, Tuple

# Import cv2 first, then patch for compatibility
import cv2

# Patch OpenCV 4.12+ compatibility - add missing constants for cv2_enumerate_cameras
if not hasattr(cv2, 'CAP_ANY'):
    cv2.CAP_ANY = 0
if not hasattr(cv2, 'CAP_V4L2'):
    cv2.CAP_V4L2 = 200
if not hasattr(cv2, 'CAP_GSTREAMER'):
    cv2.CAP_GSTREAMER = 1800
if not hasattr(cv2, 'CAP_FFMPEG'):
    cv2.CAP_FFMPEG = 1900
if not hasattr(cv2, 'CAP_MSMF'):
    cv2.CAP_MSMF = 1400

# Now import cv2_enumerate_cameras after patching
from cv2_enumerate_cameras import enumerate_cameras


def get_backend_name(backend_id: int) -> str:
    """Get backend name safely (OpenCV 4.12+ removed videoio_registry)."""
    if hasattr(cv2, 'videoio_registry'):
        return cv2.videoio_registry.getBackendName(backend_id)

    # Fallback for OpenCV 4.12+
    backend_names = {
        0: "ANY",
        200: "V4L2",
        1800: "GSTREAMER",
        1900: "FFMPEG",
        1400: "MSMF",
    }
    return backend_names.get(backend_id, f"Backend_{backend_id}")


class Camera:
    """Camera interface with continuous capture and synchronized photo capture."""

    def __init__(
        self,
        index: int = 0,
        backend: int = cv2.CAP_V4L2,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        output_dir: str = "photos",
        target_camera_name: Optional[str] = None,
        object_detector: Optional[any] = None,
        zoom_factor: float = 1.0,
    ):
        """
        Initialize camera interface.

        Args:
            index: Camera device index
            backend: OpenCV backend (cv2.CAP_V4L2 for Linux)
            width: Frame width
            height: Frame height
            fps: Target frame rate
            output_dir: Directory to save captured photos
            target_camera_name: Preferred camera name for auto-detection
            object_detector: Optional ObjectDetector for smart photo timing
            zoom_factor: Digital zoom factor (1.0 = no zoom, 2.0 = 2x zoom on center)
        """
        self.index = index
        self.backend = backend
        self.width = width
        self.height = height
        self.fps = fps
        self.output_dir = output_dir
        self.target_camera_name = target_camera_name
        self.object_detector = object_detector
        self.zoom_factor = max(1.0, zoom_factor)  # Ensure zoom is at least 1.0

        self.camera: Optional[cv2.VideoCapture] = None
        self.current_frame: Optional[any] = None
        self.frame_lock = threading.Lock()
        self.capture_thread: Optional[threading.Thread] = None
        self.capture_running = False
        self._is_initialized = False

        # Object detection state
        self.last_vehicle_bbox: Optional[Tuple[int, int, int, int]] = None
        self.vehicle_detected = False

    def probe_camera(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Probe for active camera, optionally targeting specific camera name.

        Returns:
            Tuple of (camera_index, backend) or (None, None) if not found
        """
        logging.info("Probing for active cameras...")

        cameras = enumerate_cameras(cv2.CAP_V4L2)
        logging.info(f"Found {len(cameras)} cameras on V4L2 backend:")

        for cam in cameras:
            logging.info(f"  - Index: {cam.index}, Name: '{cam.name}'")

        # If target name specified, try to find it
        if self.target_camera_name:
            for camera_info in cameras:
                if self.target_camera_name in camera_info.name:
                    logging.info(
                        f"Found target camera '{self.target_camera_name}': "
                        f"Index {camera_info.index}"
                    )
                    temp_camera = cv2.VideoCapture(camera_info.index, cv2.CAP_V4L2)
                    if temp_camera.isOpened():
                        temp_camera.release()
                        logging.info(f"Confirmed camera access at index {camera_info.index}")
                        return camera_info.index, cv2.CAP_V4L2
                    else:
                        logging.warning(f"Could not open camera at index {camera_info.index}")

        # Try any camera
        for camera_info in cameras:
            logging.info(f"Trying camera at index {camera_info.index}...")
            temp_camera = cv2.VideoCapture(camera_info.index, cv2.CAP_V4L2)
            if temp_camera.isOpened():
                temp_camera.release()
                logging.info(f"Using camera at index {camera_info.index}")
                return camera_info.index, cv2.CAP_V4L2

        logging.error("No active camera found during probing")
        return None, None

    def initialize(self) -> bool:
        """
        Initialize the camera and start continuous capture.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)

            # Try to open camera with configured settings
            logging.info(
                f"Opening camera: index={self.index}, "
                f"backend={get_backend_name(self.backend)}"
            )

            self.camera = cv2.VideoCapture(self.index, self.backend)

            # If failed, try probing
            if not self.camera.isOpened():
                logging.warning("Initial camera open failed, attempting to probe...")
                probed_index, probed_backend = self.probe_camera()

                if probed_index is not None:
                    self.index = probed_index
                    self.backend = probed_backend
                    self.camera = cv2.VideoCapture(self.index, self.backend)
                else:
                    logging.error("Could not find any working camera")
                    return False

            if not self.camera.isOpened():
                logging.error("Failed to open camera after probing")
                return False

            # Get camera name for logging
            camera_name = "Unknown Camera"
            cameras = enumerate_cameras()
            for cam_info in cameras:
                if cam_info.index == self.index:
                    camera_name = cam_info.name
                    break

            logging.info(
                f"Successfully opened camera: '{camera_name}' "
                f"(Index: {self.index}, Backend: {get_backend_name(self.backend)})"
            )

            # Optimize camera settings
            logging.info("Configuring camera settings...")
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for fresh frames
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)

            # Log digital zoom configuration
            if self.zoom_factor > 1.0:
                logging.info(f"Digital zoom enabled: {self.zoom_factor}x (center crop)")
            else:
                logging.info("Digital zoom disabled (zoom_factor: 1.0)")

            # Warm up camera
            logging.info("Warming up camera...")
            for _ in range(3):
                self.camera.read()

            # Start continuous capture thread
            self._start_continuous_capture()

            self._is_initialized = True
            logging.info("Camera initialized successfully")
            return True

        except Exception as e:
            logging.error(f"Failed to initialize camera: {e}")
            return False

    def _apply_digital_zoom(self, frame):
        """
        Apply digital zoom by cropping center region and resizing.

        Args:
            frame: Input frame

        Returns:
            Zoomed frame (same dimensions as input)
        """
        if self.zoom_factor <= 1.0:
            return frame

        height, width = frame.shape[:2]

        # Calculate crop dimensions (center crop)
        crop_width = int(width / self.zoom_factor)
        crop_height = int(height / self.zoom_factor)

        # Calculate crop coordinates (centered)
        x1 = (width - crop_width) // 2
        y1 = (height - crop_height) // 2
        x2 = x1 + crop_width
        y2 = y1 + crop_height

        # Crop center region
        cropped = frame[y1:y2, x1:x2]

        # Resize back to original dimensions
        zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

        return zoomed

    def _continuous_capture_worker(self):
        """Worker function for continuous camera capture in separate thread."""
        while self.capture_running:
            if self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret:
                    # Apply digital zoom if enabled
                    if self.zoom_factor > 1.0:
                        frame = self._apply_digital_zoom(frame)

                    with self.frame_lock:
                        self.current_frame = frame.copy()
                else:
                    logging.warning("Failed to read frame in continuous capture")
                    time.sleep(0.01)
            else:
                logging.warning("Camera not available in continuous capture")
                time.sleep(0.1)

            # Small delay to prevent overwhelming the system
            time.sleep(0.001)

    def _start_continuous_capture(self):
        """Start the continuous camera capture thread."""
        if self.capture_running:
            logging.warning("Continuous capture already running")
            return

        self.capture_running = True
        self.capture_thread = threading.Thread(
            target=self._continuous_capture_worker, daemon=True
        )
        self.capture_thread.start()
        logging.info("Continuous camera capture started")

    def _stop_continuous_capture(self):
        """Stop the continuous camera capture thread."""
        if not self.capture_running:
            return

        self.capture_running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        logging.info("Continuous camera capture stopped")

    def capture_photo(self, timestamp: float, suffix: str = "") -> Optional[str]:
        """
        Capture a single photo from current frame buffer.

        Args:
            timestamp: Event timestamp (Unix timestamp)
            suffix: Optional suffix for filename

        Returns:
            Path to saved image file, or None if capture failed
        """
        if not self._is_initialized:
            logging.error("Camera not initialized")
            return None

        with self.frame_lock:
            if self.current_frame is None:
                logging.error("No current frame available")
                return None
            frame = self.current_frame.copy()

        # Format filename: car_YYYY_MM_DD_HH-MM-SS.jpg
        dt = datetime.fromtimestamp(timestamp)
        base_filename = dt.strftime("car_%Y_%m_%d_%H-%M-%S")
        filename = f"{base_filename}{suffix}.jpg"
        filepath = os.path.join(self.output_dir, filename)

        # Save image in separate thread to avoid blocking
        threading.Thread(
            target=self._save_image, args=(frame, filepath), daemon=True
        ).start()

        return filepath

    def _save_image(self, frame, filepath: str):
        """Helper to save image in background thread."""
        try:
            cv2.imwrite(filepath, frame)
            logging.debug(f"Image saved: {filepath}")
        except Exception as e:
            logging.error(f"Error saving image {filepath}: {e}")

    def capture_sequence(
        self, timestamp: float, count: int = 4, interval: float = 0.25
    ) -> list[str]:
        """
        Capture a sequence of photos with timing intervals.

        Args:
            timestamp: Base timestamp for event
            count: Number of photos to capture
            interval: Time interval between captures in seconds

        Returns:
            List of saved photo paths
        """
        photos = []

        # Immediate capture
        photo_path = self.capture_photo(timestamp, suffix="-0")
        if photo_path:
            photos.append(photo_path)

        # Delayed captures
        def delayed_capture():
            for i in range(1, count):
                time.sleep(interval)
                photo_path = self.capture_photo(timestamp, suffix=f"-{i}")
                if photo_path:
                    photos.append(photo_path)

        threading.Thread(target=delayed_capture, daemon=True).start()

        return photos

    def get_current_frame(self) -> Optional[any]:
        """
        Get the current frame from the capture buffer.

        Returns:
            Current frame (numpy array) or None if not available
        """
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None

    def close(self):
        """Release camera resources."""
        self._stop_continuous_capture()

        if self.camera:
            self.camera.release()
            self._is_initialized = False
            logging.info("Camera released")

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
