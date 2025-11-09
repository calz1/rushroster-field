"""
Async frame processor that runs YOLO detection on burst-captured frames.
Runs in a separate process to avoid blocking the main detection loop.
"""

import logging
import cv2
import multiprocessing
import time
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
import shutil


@dataclass
class FrameAnalysis:
    """Analysis result for a single frame."""
    frame_path: str
    detected: bool
    bbox: Optional[Tuple[int, int, int, int]]
    centering_score: float
    confidence: float
    class_name: Optional[str]


@dataclass
class ProcessingResult:
    """Result of processing all frames for an event."""
    event_id: int
    best_frame_path: Optional[str]
    total_frames: int
    frames_with_detection: int
    processing_time: float
    best_score: float


def _calculate_centering_score(
    bbox: Tuple[int, int, int, int],
    frame_width: int,
    frame_height: int
) -> float:
    """
    Calculate how well-centered an object is in the frame.

    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels

    Returns:
        Score from 0-1, where 1.0 is perfectly centered
    """
    x1, y1, x2, y2 = bbox

    # Get bbox center
    bbox_center_x = (x1 + x2) / 2
    bbox_center_y = (y1 + y2) / 2

    # Get frame center
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2

    # Calculate distance from center (normalized)
    x_distance = abs(bbox_center_x - frame_center_x) / (frame_width / 2)
    y_distance = abs(bbox_center_y - frame_center_y) / (frame_height / 2)

    # Combined distance (0 = perfect center, 1 = edge)
    distance = (x_distance + y_distance) / 2

    # Convert to score (1 = perfect center, 0 = edge)
    score = 1.0 - distance

    # Also consider bbox size (larger is better, vehicle is closer)
    bbox_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_width * frame_height
    size_ratio = bbox_area / frame_area

    # Combine centering and size (70% centering, 30% size)
    # Size should be between 5-40% of frame for good shot
    optimal_size_ratio = 0.20  # 20% is ideal
    size_score = 1.0 - min(abs(size_ratio - optimal_size_ratio) / optimal_size_ratio, 1.0)

    final_score = (score * 0.7) + (size_score * 0.3)

    return final_score


def _process_frames_worker(
    event_id: int,
    frame_paths: List[str],
    output_dir: str,
    temp_dir: str,
    model_dir: str,
    model_size: str,
    confidence_threshold: float,
    detection_mode: str,
    inference_size: int,
    nms_threshold: float,
) -> ProcessingResult:
    """
    Worker function that runs in a separate process to analyze frames.
    This is the function that gets spawned in a subprocess.

    Args:
        event_id: Event ID
        frame_paths: List of frame file paths to analyze
        output_dir: Directory to save the best frame
        temp_dir: Temporary directory containing burst frames
        model_dir: Directory containing YOLO model files
        model_size: YOLO model size
        confidence_threshold: Detection confidence threshold
        detection_mode: Detection mode preset
        inference_size: YOLO inference size
        nms_threshold: Non-maximum suppression threshold

    Returns:
        ProcessingResult with best frame selection
    """
    # Import here to avoid issues with multiprocessing
    import sys
    from pathlib import Path

    # Ensure the parent module can be imported in subprocess
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.sensors.object_detector import ObjectDetector

    start_time = time.time()

    # Initialize object detector in this subprocess
    detector = ObjectDetector(
        model_size=model_size,
        confidence_threshold=confidence_threshold,
        detection_mode=detection_mode,
        inference_size=inference_size,
        nms_threshold=nms_threshold,
        model_dir=model_dir,
    )

    if not detector.initialize():
        logging.error("Failed to initialize object detector in subprocess")
        return ProcessingResult(
            event_id=event_id,
            best_frame_path=None,
            total_frames=len(frame_paths),
            frames_with_detection=0,
            processing_time=time.time() - start_time,
            best_score=0.0
        )

    # Analyze each frame
    analyses = []

    for frame_path in frame_paths:
        try:
            # Load frame
            frame = cv2.imread(frame_path)
            if frame is None:
                logging.warning(f"Failed to load frame: {frame_path}")
                continue

            height, width = frame.shape[:2]

            # Run detection
            detected, bbox, metadata = detector.detect_with_metadata(frame)

            if detected and bbox:
                # Calculate centering score
                centering_score = _calculate_centering_score(bbox, width, height)

                analyses.append(FrameAnalysis(
                    frame_path=frame_path,
                    detected=True,
                    bbox=bbox,
                    centering_score=centering_score,
                    confidence=metadata.get('confidence', 0.0),
                    class_name=metadata.get('class_name', 'unknown')
                ))

                logging.debug(
                    f"Frame {Path(frame_path).name}: {metadata.get('class_name')} "
                    f"detected, centering_score={centering_score:.2f}, "
                    f"confidence={metadata.get('confidence', 0.0):.2f}"
                )
            else:
                analyses.append(FrameAnalysis(
                    frame_path=frame_path,
                    detected=False,
                    bbox=None,
                    centering_score=0.0,
                    confidence=0.0,
                    class_name=None
                ))

        except Exception as e:
            logging.error(f"Error processing frame {frame_path}: {e}")
            continue

    # Find best frame (highest centering score)
    best_frame = None
    best_score = 0.0
    frames_with_detection = sum(1 for a in analyses if a.detected)

    if analyses:
        # Filter to only frames with detections
        detected_frames = [a for a in analyses if a.detected]

        if detected_frames:
            best_frame = max(detected_frames, key=lambda x: x.centering_score)
            best_score = best_frame.centering_score
        else:
            # No detections found, use middle frame as fallback
            middle_idx = len(analyses) // 2
            best_frame = analyses[middle_idx]
            logging.warning(f"No detections in any frame for event {event_id}, using middle frame as fallback")

    # Copy best frame to final location
    best_frame_path = None
    if best_frame:
        try:
            # Generate final filename
            from datetime import datetime
            timestamp = datetime.now()
            final_filename = timestamp.strftime("car_%Y_%m_%d_%H-%M-%S.jpg")
            final_path = Path(output_dir) / final_filename

            # Copy best frame to final location
            shutil.copy2(best_frame.frame_path, final_path)
            # Store absolute path in database to work regardless of current working directory
            best_frame_path = str(final_path.resolve())

            logging.info(
                f"Event {event_id}: Selected best frame with score {best_score:.2f} "
                f"({frames_with_detection}/{len(frame_paths)} frames had detections)"
            )

        except Exception as e:
            logging.error(f"Failed to save best frame for event {event_id}: {e}")

    # Clean up temp directory
    try:
        shutil.rmtree(temp_dir)
        logging.debug(f"Cleaned up temp directory: {temp_dir}")
    except Exception as e:
        logging.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")

    processing_time = time.time() - start_time

    return ProcessingResult(
        event_id=event_id,
        best_frame_path=best_frame_path,
        total_frames=len(frame_paths),
        frames_with_detection=frames_with_detection,
        processing_time=processing_time,
        best_score=best_score
    )


class AsyncFrameProcessor:
    """
    Manages async processing of burst-captured frames using multiprocessing.
    Spawns subprocesses to run YOLO detection without blocking main thread.
    """

    def __init__(
        self,
        output_dir: str = "photos",
        model_dir: str = "darknet_models",
        model_size: str = "tiny",
        confidence_threshold: float = 0.5,
        detection_mode: str = "all_moving",
        inference_size: int = 320,
        nms_threshold: float = 0.4,
    ):
        """
        Initialize async frame processor.

        Args:
            output_dir: Directory to save final selected frames
            model_dir: Directory containing YOLO model files
            model_size: YOLO model size
            confidence_threshold: Detection confidence threshold
            detection_mode: Detection mode preset
            inference_size: YOLO inference size
            nms_threshold: Non-maximum suppression threshold
        """
        self.output_dir = output_dir
        self.model_dir = model_dir
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.detection_mode = detection_mode
        self.inference_size = inference_size
        self.nms_threshold = nms_threshold

        # Track active processes
        self.active_processes = {}

        logging.info(f"Async frame processor initialized: output_dir={output_dir}")

    def process_burst_async(
        self,
        event_id: int,
        frame_paths: List[str],
        temp_dir: str,
        callback: Optional[callable] = None
    ) -> multiprocessing.Process:
        """
        Start async processing of burst frames in a separate process.

        Args:
            event_id: Event ID
            frame_paths: List of frame paths to process
            temp_dir: Temporary directory containing frames
            callback: Optional callback function to call with result

        Returns:
            Process handle
        """
        logging.info(f"Starting async processing for event {event_id} ({len(frame_paths)} frames)")

        # Create process
        process = multiprocessing.Process(
            target=self._process_and_callback,
            args=(
                event_id,
                frame_paths,
                self.output_dir,
                temp_dir,
                self.model_dir,
                self.model_size,
                self.confidence_threshold,
                self.detection_mode,
                self.inference_size,
                self.nms_threshold,
                callback,
            )
        )

        process.start()
        self.active_processes[event_id] = process

        logging.debug(f"Spawned processing subprocess for event {event_id} (PID: {process.pid})")

        return process

    def _process_and_callback(self, *args):
        """Wrapper to run worker and handle callback."""
        callback = args[-1]
        worker_args = args[:-1]

        result = _process_frames_worker(*worker_args)

        if callback:
            callback(result)

    def wait_for_event(self, event_id: int, timeout: Optional[float] = None):
        """
        Wait for processing of a specific event to complete.

        Args:
            event_id: Event ID to wait for
            timeout: Optional timeout in seconds
        """
        if event_id in self.active_processes:
            process = self.active_processes[event_id]
            process.join(timeout=timeout)

            if process.is_alive():
                logging.warning(f"Processing for event {event_id} timed out")
            else:
                del self.active_processes[event_id]

    def cleanup_finished_processes(self):
        """Clean up finished processes from tracking dict."""
        finished = [eid for eid, proc in self.active_processes.items() if not proc.is_alive()]

        for event_id in finished:
            del self.active_processes[event_id]
            logging.debug(f"Cleaned up finished process for event {event_id}")

    def get_active_count(self) -> int:
        """Get count of active processing jobs."""
        self.cleanup_finished_processes()
        return len(self.active_processes)
