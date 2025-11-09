"""
Object detection module using YOLOv4-tiny for vehicle detection.
Optimized for Raspberry Pi 4 with low-latency inference using OpenCV DNN.
"""

import logging
import time
import cv2
from typing import Optional, Tuple, List
from pathlib import Path
import numpy as np


class ObjectDetector:
    """
    YOLOv4-tiny object detector optimized for vehicle detection.
    Uses OpenCV DNN backend for CPU-optimized inference on Raspberry Pi 4.
    """

    # COCO classes for detection (same IDs as RF-DETR)
    ALL_TARGET_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    # Preset detection modes
    DETECTION_PRESETS = {
        "vehicles": [2, 3, 5, 7],  # cars, motorcycles, buses, trucks
        "all_moving": [0, 1, 2, 3, 5, 7],  # people, bicycles, and vehicles
        "pedestrians": [0],  # people only
        "bikes": [1, 3],  # bicycles and motorcycles
    }

    def __init__(
        self,
        model_size: str = "tiny",
        confidence_threshold: float = 0.5,
        detection_mode: str = "all_moving",
        custom_classes: Optional[List[int]] = None,
        inference_size: int = 320,  # Input size for YOLO (320 = fast, 416 = accurate)
        nms_threshold: float = 0.4,  # Non-maximum suppression threshold
        model_dir: Optional[str] = None,
    ):
        """
        Initialize object detector.

        Args:
            model_size: Model size ("tiny" for YOLOv4-tiny, fastest)
            confidence_threshold: Minimum confidence for detections
            detection_mode: Preset mode (vehicles, all_moving, pedestrians, bikes)
            custom_classes: Custom list of COCO class IDs to detect (overrides mode)
            inference_size: Input size for inference (320=fast, 416=accurate)
            nms_threshold: Non-maximum suppression threshold
            model_dir: Directory containing model files (default: darknet_models)
        """
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.inference_size = inference_size
        self.nms_threshold = nms_threshold
        self.model_dir = Path(model_dir) if model_dir else Path("darknet_models")

        self.net = None
        self.class_names = []
        self.output_layers = []
        self._is_initialized = False

        # Set detection classes
        if custom_classes is not None:
            self.target_classes = custom_classes
        elif detection_mode in self.DETECTION_PRESETS:
            self.target_classes = self.DETECTION_PRESETS[detection_mode]
        else:
            logging.warning(f"Unknown detection mode '{detection_mode}', using 'all_moving'")
            self.target_classes = self.DETECTION_PRESETS["all_moving"]

        logging.info(
            f"Detection mode: {detection_mode}, "
            f"classes: {[self.ALL_TARGET_CLASSES.get(c, c) for c in self.target_classes]}"
        )

        # Performance tracking
        self.last_inference_time = 0.0
        self.avg_inference_time = 0.0
        self._inference_count = 0

    def _download_model_files(self) -> bool:
        """Download YOLOv4-tiny model files if not present."""
        self.model_dir.mkdir(exist_ok=True)

        cfg_file = "yolov4-tiny.cfg"
        weights_file = "yolov4-tiny.weights"
        names_file = "coco.names"

        cfg_path = self.model_dir / cfg_file
        weights_path = self.model_dir / weights_file
        names_path = self.model_dir / names_file

        base_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master"

        files_to_download = [
            (cfg_path, f"{base_url}/cfg/{cfg_file}"),
            (
                weights_path,
                "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",
            ),
            (names_path, f"{base_url}/cfg/{names_file}"),
        ]

        for file_path, url in files_to_download:
            if not file_path.exists():
                logging.info(f"Downloading {file_path.name}...")
                try:
                    import urllib.request

                    urllib.request.urlretrieve(url, file_path)
                    logging.info(f"Downloaded {file_path.name}")
                except Exception as e:
                    logging.error(f"Failed to download {file_path.name}: {e}")
                    logging.info(f"Please manually download from: {url}")
                    return False

        return True

    def _load_class_names(self) -> List[str]:
        """Load COCO class names."""
        names_path = self.model_dir / "coco.names"

        if not names_path.exists():
            logging.error(f"Class names file not found: {names_path}")
            return []

        with open(names_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]

        logging.info(f"Loaded {len(classes)} class names from COCO")
        return classes

    def initialize(self) -> bool:
        """
        Initialize the YOLO model using OpenCV DNN.

        Returns:
            True if initialization successful
        """
        try:
            logging.info(f"Initializing YOLOv4-{self.model_size} model with OpenCV DNN...")

            cfg_path = self.model_dir / f"yolov4-{self.model_size}.cfg"
            weights_path = self.model_dir / f"yolov4-{self.model_size}.weights"

            # Download model files if not present
            if not cfg_path.exists() or not weights_path.exists():
                logging.info("Model files not found, downloading...")
                if not self._download_model_files():
                    return False

            # Load network
            self.net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))

            # Set computation backend to CPU (DNN_BACKEND_OPENCV is CPU-optimized)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # Get output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [
                layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()
            ]

            logging.info(f"Model loaded. Output layers: {self.output_layers}")

            # Load class names
            self.class_names = self._load_class_names()

            self._is_initialized = True
            logging.info(
                f"YOLOv4-{self.model_size} initialized successfully "
                f"(inference size: {self.inference_size}x{self.inference_size})"
            )
            return True

        except Exception as e:
            logging.error(f"Failed to initialize object detector: {e}")
            return False

    def detect_vehicles(
        self, frame: np.ndarray
    ) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """
        Detect target objects (vehicles, people, bikes) in frame and return best bounding box.

        Args:
            frame: Image frame (BGR format from OpenCV)

        Returns:
            Tuple of (object_detected, best_bbox)
            best_bbox is (x1, y1, x2, y2) or None
        """
        detected, bbox, metadata = self.detect_with_metadata(frame)
        return detected, bbox

    def detect_with_metadata(
        self, frame: np.ndarray
    ) -> Tuple[bool, Optional[Tuple[int, int, int, int]], Optional[dict]]:
        """
        Detect target objects and return full detection metadata.

        Args:
            frame: Image frame (BGR format from OpenCV)

        Returns:
            Tuple of (object_detected, best_bbox, metadata)
            metadata includes class_name, confidence, inference_time
        """
        if not self._is_initialized or self.net is None:
            return False, None, None

        try:
            start_time = time.time()

            height, width = frame.shape[:2]

            # Create blob from image (YOLO expects normalized RGB)
            blob = cv2.dnn.blobFromImage(
                frame,
                1 / 255.0,
                (self.inference_size, self.inference_size),
                swapRB=True,
                crop=False,
            )

            # Set input and run inference
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)

            # Track inference time
            inference_time = time.time() - start_time
            self.last_inference_time = inference_time
            self._inference_count += 1
            self.avg_inference_time = (
                self.avg_inference_time * (self._inference_count - 1) + inference_time
            ) / self._inference_count

            # Parse detections
            class_ids = []
            confidences = []
            boxes = []

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > self.confidence_threshold:
                        # YOLO returns center x, center y, width, height (normalized)
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Convert to top-left corner
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply non-maximum suppression
            indices = cv2.dnn.NMSBoxes(
                boxes, confidences, self.confidence_threshold, self.nms_threshold
            )

            # Filter for target classes only
            target_detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    class_id = class_ids[i]
                    if class_id in self.target_classes:
                        confidence = confidences[i]
                        x, y, w, h = boxes[i]

                        # Convert to (x1, y1, x2, y2) format
                        bbox = (x, y, x + w, y + h)

                        class_name = self.ALL_TARGET_CLASSES.get(
                            class_id,
                            self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                        )

                        target_detections.append({
                            'bbox': bbox,
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name
                        })

            # If targets detected, return the one with highest confidence
            if target_detections:
                best_detection = max(target_detections, key=lambda x: x['confidence'])
                bbox = best_detection['bbox']

                logging.info(
                    f"Object detected: {best_detection['class_name']} "
                    f"(confidence: {best_detection['confidence']:.2f}, "
                    f"inference: {inference_time*1000:.1f}ms)"
                )

                metadata = {
                    'class_name': best_detection['class_name'],
                    'confidence': best_detection['confidence'],
                    'inference_time_ms': inference_time * 1000,
                }

                return True, tuple(map(int, bbox)), metadata

            return False, None, None

        except Exception as e:
            logging.error(f"Error during object detection: {e}")
            return False, None, None

    def get_centered_bbox(
        self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]
    ) -> bool:
        """
        Check if bounding box is well-centered in frame.
        Useful for determining optimal photo timing.

        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            frame_shape: Frame dimensions (height, width)

        Returns:
            True if bbox is centered (good for photo)
        """
        if bbox is None:
            return False

        x1, y1, x2, y2 = bbox
        height, width = frame_shape[:2]

        # Calculate bbox center
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2

        # Calculate frame center
        frame_center_x = width / 2
        frame_center_y = height / 2

        # Check if bbox center is within 25% of frame center
        x_threshold = width * 0.25
        y_threshold = height * 0.25

        x_centered = abs(bbox_center_x - frame_center_x) < x_threshold
        y_centered = abs(bbox_center_y - frame_center_y) < y_threshold

        return x_centered and y_centered

    def get_bbox_size(self, bbox: Tuple[int, int, int, int]) -> int:
        """
        Calculate bounding box area.

        Args:
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Area in pixels
        """
        if bbox is None:
            return 0

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return width * height

    def get_performance_stats(self) -> dict:
        """
        Get performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        return {
            "last_inference_ms": self.last_inference_time * 1000,
            "avg_inference_ms": self.avg_inference_time * 1000,
            "inference_count": self._inference_count,
            "model_size": self.model_size,
        }

    def annotate_frame(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Annotate frame with bounding box and detection info (for debugging/visualization).

        Args:
            frame: Image frame to annotate
            bbox: Bounding box to draw (x1, y1, x2, y2)
            metadata: Detection metadata with class_name, confidence, etc.

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        if bbox is not None:
            x1, y1, x2, y2 = bbox

            # Draw bounding box with thicker line
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(annotated, (center_x, center_y), 5, (0, 0, 255), -1)

            # Add label with class name and confidence if metadata provided
            if metadata:
                class_name = metadata.get("class_name", "object")
                confidence = metadata.get("confidence", 0)
                label = f"{class_name}: {confidence:.2f}"

                # Calculate label size for background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Draw label background
                label_y = max(y1 - 10, label_height + 10)
                cv2.rectangle(
                    annotated,
                    (x1, label_y - label_height - baseline - 5),
                    (x1 + label_width + 10, label_y + baseline),
                    (0, 255, 0),
                    -1,
                )

                # Draw label text
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 5, label_y - 5),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                )

        return annotated

    def close(self):
        """Release resources."""
        self._is_initialized = False
        self.net = None
        logging.info("Object detector closed")
