"""Camera frame streaming and visualization endpoints."""

import logging
import time as time_module
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from ..hud import render_hud_overlay


def register_camera_routes(
    app: FastAPI,
    camera: Any,
    detection_module: Any,
    object_detector: Any,
    config: Any,
    dashboard_overlays: bool = False,
) -> None:
    """
    Register camera frame endpoint.

    Args:
        app: FastAPI application instance
        camera: Camera instance for live feed (optional)
        detection_module: Detection module for live status (optional)
        object_detector: Object detector instance for detection overlays (optional)
        config: Configuration object
        dashboard_overlays: Enable object detection overlays on dashboard (optional)
    """

    # Cache for annotated frames to prevent running detection on every request
    cache_data = {
        "last_annotated_frame": None,
        "last_annotation_time": 0,
        "annotation_cache_duration": 1.0,  # Cache annotated frames for 1 second (reduces Pi CPU load)
    }

    @app.get("/camera/frame")
    async def get_camera_frame(detections: bool = True, hud: bool = True):
        """
        Get current camera frame as JPEG, optionally with object detection overlays and HUD.

        Args:
            detections: Whether to overlay detection boxes (default: True)
            hud: Whether to render HUD overlay (default: True)
        """
        try:
            if not camera or not camera._is_initialized:
                raise HTTPException(status_code=503, detail="Camera not available")

            # Get current frame from camera
            frame = camera.get_current_frame()
            if frame is None:
                raise HTTPException(status_code=503, detail="No frame available")

            import cv2

            # Downscale frame for web streaming (significantly reduces bandwidth and encoding time)
            # Target size matches object detector inference size for consistency
            target_width = 640  # Good balance between quality and performance
            original_height, original_width = frame.shape[:2]
            scale = target_width / original_width
            target_height = int(original_height * scale)

            # Downscale the frame
            frame_downscaled = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

            # Run object detection and annotate frame if enabled
            # Use cached result if recent to avoid overloading the Pi
            current_time = time_module.time()

            if (detections and
                dashboard_overlays and
                object_detector and
                object_detector._is_initialized):
                # Check if we should run detection (not too recent)
                if current_time - cache_data["last_annotation_time"] >= cache_data["annotation_cache_duration"]:
                    try:
                        # Run detection on ORIGINAL frame for accuracy
                        detected, bbox, metadata = object_detector.detect_with_metadata(frame)
                        if detected and bbox:
                            # Scale bbox coordinates down to match downscaled frame
                            x1, y1, x2, y2 = bbox
                            scaled_bbox = (
                                int(x1 * scale),
                                int(y1 * scale),
                                int(x2 * scale),
                                int(y2 * scale)
                            )
                            # Annotate the downscaled frame with scaled bbox
                            frame_downscaled = object_detector.annotate_frame(frame_downscaled, scaled_bbox, metadata)
                        # Cache the annotated downscaled frame
                        cache_data["last_annotated_frame"] = frame_downscaled
                        cache_data["last_annotation_time"] = current_time
                    except Exception as e:
                        logging.debug(f"Detection overlay failed: {e}")
                        # Continue with unannotated frame
                elif cache_data["last_annotated_frame"] is not None:
                    # Use cached annotated frame
                    frame_downscaled = cache_data["last_annotated_frame"]

            # Render HUD overlay if enabled and HUD is configured
            if hud and hasattr(config, 'hud') and config.hud.enabled:
                try:
                    # Get live sensor data for HUD
                    current_speed = None
                    tracking = False

                    if detection_module:
                        stats = detection_module.get_statistics()
                        current_speed = stats.get('recent_speed', None)
                        tracking = stats.get('tracking', False)

                    # Get speed limit from config
                    speed_limit = config.device.speed_limit

                    # Get HUD display configuration
                    hud_config = {
                        'show_speed': config.hud.show_speed,
                        'show_speed_limit': config.hud.show_speed_limit,
                        'show_tracking_state': config.hud.show_tracking_state,
                    }

                    # Render HUD
                    frame_downscaled = render_hud_overlay(
                        frame_downscaled,
                        current_speed,
                        speed_limit,
                        tracking,
                        hud_config
                    )
                except Exception as e:
                    logging.debug(f"HUD rendering failed: {e}")
                    # Continue without HUD if rendering fails

            # Encode downscaled frame as JPEG with good compression
            success, buffer = cv2.imencode('.jpg', frame_downscaled, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                raise HTTPException(status_code=500, detail="Failed to encode frame")

            # Return JPEG image
            return Response(content=buffer.tobytes(), media_type="image/jpeg")

        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Error getting camera frame: {e}")
            raise HTTPException(status_code=500, detail=str(e))
