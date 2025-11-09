#!/usr/bin/env python3
"""
Main entry point for RushRoster field device.
Initializes and runs all components: radar, camera, detection, API, and upload service.
"""

import sys
import logging
import signal
import threading
import uvicorn
import socket
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import ConfigLoader
from src.sensors import RadarSensor, Camera, ObjectDetector
from src.database import SQLiteManager
from src.detection import SyncDetectionModule, BurstDetectionModule
from src.api import create_api
from src.upload import SyncService


# Global flag for graceful shutdown
shutdown_event = threading.Event()


def setup_logging(config=None, level=logging.INFO):
    """
    Configure logging for the application.

    Args:
        config: Configuration object with logging settings
        level: Logging level (default: logging.INFO)
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    max_bytes = 10485760  # 10MB default
    backup_count = 5  # Default backup count

    if config and hasattr(config, 'logging'):
        if config.logging.enabled:
            log_dir = Path(config.logging.log_dir)
            max_bytes = config.logging.rotation.max_bytes
            backup_count = config.logging.rotation.backup_count
        else:
            # If logging disabled, use minimal console-only setup
            logging.basicConfig(
                level=level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[logging.StreamHandler(sys.stdout)],
            )
            return

    # Create log directory if rotation is enabled
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log file path
    log_file = log_dir / "rush-roster.log"

    # Create rotating file handler
    rotating_handler = RotatingFileHandler(
        str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
    )

    # Set up logging with both console and rotating file handlers
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            rotating_handler,
        ],
    )


def get_local_ip():
    """Get the local network IP address."""
    try:
        # Create a socket to determine the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Connect to a public DNS server (doesn't actually send data)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logging.info(f"Received signal {signum}, initiating shutdown...")
    shutdown_event.set()


def main():
    """Main application entry point."""
    # First, do basic logging setup without config
    setup_logging(level=logging.INFO)
    logging.info("=" * 60)
    logging.info("Starting RushRoster Field Device")
    logging.info("=" * 60)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Load configuration
    logging.info("Loading configuration...")
    config_loader = ConfigLoader("config.yaml")
    config = config_loader.load()

    # Reconfigure logging with loaded config settings
    setup_logging(config=config, level=logging.INFO)

    logging.info(f"Device ID: {config.device.id}")
    logging.info(f"Location: {config.device.street_name}")
    logging.info(f"Speed Limit: {config.device.speed_limit} MPH")

    # Initialize database
    logging.info("Initializing database...")
    db = SQLiteManager("data/events.db")
    db.initialize_database()

    # Update device metadata
    db.update_metadata(
        device_id=config.device.id,
        latitude=config.device.latitude,
        longitude=config.device.longitude,
        street_name=config.device.street_name,
        speed_limit=config.device.speed_limit,
    )

    # Initialize sensors
    logging.info("Initializing sensors...")

    radar_cfg = config.sensors.get("radar")
    camera_cfg = config.sensors.get("camera")

    radar = RadarSensor(
        port=radar_cfg.port,
        speed_limit=config.device.speed_limit,
        min_speed=config.thresholds.min_speed,
        max_speed=config.thresholds.max_speed,
        direction="inbound",  # Track only inbound vehicles
    )

    camera = Camera(
        index=camera_cfg.device,
        width=camera_cfg.resolution[0],
        height=camera_cfg.resolution[1],
        output_dir="photos",
        target_camera_name=camera_cfg.target_name,
        zoom_factor=camera_cfg.zoom_factor,
    )

    # Initialize sensors (optional - run in demo mode if hardware not available)
    # Use timeouts to prevent hanging on initialization
    logging.info("Initializing radar sensor (timeout: 5s)...")
    radar_ready = False
    try:
        def init_radar():
            nonlocal radar_ready
            radar_ready = radar.initialize()

        radar_thread = threading.Thread(target=init_radar)
        radar_thread.start()
        radar_thread.join(timeout=5.0)

        if radar_thread.is_alive():
            logging.warning("Radar initialization timed out")
            radar_ready = False
    except Exception as e:
        logging.warning(f"Radar initialization failed: {e}")
        radar_ready = False

    if not radar_ready:
        logging.warning("Failed to initialize radar sensor - running in demo mode")

    logging.info("Initializing camera (timeout: 5s)...")
    camera_ready = False
    try:
        def init_camera():
            nonlocal camera_ready
            camera_ready = camera.initialize()

        camera_thread = threading.Thread(target=init_camera)
        camera_thread.start()
        camera_thread.join(timeout=5.0)

        if camera_thread.is_alive():
            logging.warning("Camera initialization timed out")
            camera_ready = False
    except Exception as e:
        logging.warning(f"Camera initialization failed: {e}")
        camera_ready = False

    if not camera_ready:
        logging.warning("Failed to initialize camera - running in demo mode")

    # Initialize object detector (optional, for smart photo capture and continuous detection)
    detector_cfg = config.sensors.get("object_detector", {})
    detector_enabled = detector_cfg.get("enabled", True) if isinstance(detector_cfg, dict) else True
    detector_ready = False
    continuous_detection = False
    detection_interval = 0.3
    dashboard_overlays = False

    if detector_enabled and camera_ready:
        model_size = detector_cfg.get("model_size", "tiny") if isinstance(detector_cfg, dict) else "tiny"
        confidence = detector_cfg.get("confidence_threshold", 0.5) if isinstance(detector_cfg, dict) else 0.5
        detection_mode = detector_cfg.get("detection_mode", "all_moving") if isinstance(detector_cfg, dict) else "all_moving"
        inference_size = detector_cfg.get("inference_size", 320) if isinstance(detector_cfg, dict) else 320
        nms_threshold = detector_cfg.get("nms_threshold", 0.4) if isinstance(detector_cfg, dict) else 0.4
        continuous_detection = detector_cfg.get("continuous_detection", True) if isinstance(detector_cfg, dict) else True
        detection_interval = detector_cfg.get("detection_interval", 0.3) if isinstance(detector_cfg, dict) else 0.3
        dashboard_overlays = detector_cfg.get("dashboard_overlays", False) if isinstance(detector_cfg, dict) else False

        logging.info(f"Initializing object detector (YOLOv4-{model_size}, mode: {detection_mode}, size: {inference_size})...")
        if continuous_detection:
            logging.info(f"Continuous detection enabled (interval: {detection_interval}s)")
        if dashboard_overlays:
            logging.info("Dashboard detection overlays enabled")
        else:
            logging.info("Dashboard detection overlays disabled (raw camera feed only)")

        object_detector = ObjectDetector(
            model_size=model_size,
            confidence_threshold=confidence,
            detection_mode=detection_mode,
            inference_size=inference_size,
            nms_threshold=nms_threshold
        )

        try:
            detector_ready = object_detector.initialize()
            if detector_ready:
                logging.info("Object detector initialized - smart capture enabled")
            else:
                logging.warning("Object detector initialization failed - using basic capture")
        except Exception as e:
            logging.warning(f"Object detector initialization failed: {e}")
            detector_ready = False
    else:
        if not detector_enabled:
            logging.info("Object detector disabled in config")
        else:
            logging.info("Skipping object detector - camera not available")
        object_detector = None

    # Check if we should run in demo mode
    demo_mode = not (radar_ready and camera_ready)
    if demo_mode:
        logging.warning("=" * 60)
        logging.warning("Running in DEMO MODE - hardware not available")
        logging.warning("API and dashboard will work, but detection is disabled")
        logging.warning("=" * 60)

    # Create detection module (only if not in demo mode)
    detection = None
    detection_thread = None

    if not demo_mode:
        logging.info("Starting detection module...")

        # Use BurstDetectionModule for optimized Raspberry Pi performance
        # Captures burst of frames, then processes with YOLO asynchronously
        if detector_enabled:
            logging.info("Using BurstDetectionModule with async YOLO processing")
            detection = BurstDetectionModule(
                radar=radar,
                camera=camera,
                db=db,
                object_detector_config={
                    'model_dir': 'darknet_models',
                    'model_size': detector_cfg.get("model_size", "tiny") if isinstance(detector_cfg, dict) else "tiny",
                    'confidence_threshold': detector_cfg.get("confidence_threshold", 0.5) if isinstance(detector_cfg, dict) else 0.5,
                    'detection_mode': detector_cfg.get("detection_mode", "all_moving") if isinstance(detector_cfg, dict) else "all_moving",
                    'inference_size': detector_cfg.get("inference_size", 320) if isinstance(detector_cfg, dict) else 320,
                    'nms_threshold': detector_cfg.get("nms_threshold", 0.4) if isinstance(detector_cfg, dict) else 0.4,
                },
                speed_limit=config.device.speed_limit,
                photo_trigger_threshold=config.thresholds.photo_trigger_over_limit,
                burst_capture_interval=0.1,  # 100ms between frames
                burst_max_frames=20,  # Capture max 20 frames (2 seconds)
            )
        else:
            # Fallback to old module if detector disabled
            logging.info("Using SyncDetectionModule (object detector disabled)")
            detection = SyncDetectionModule(
                radar=radar,
                camera=camera,
                db=db,
                object_detector=None,
                speed_limit=config.device.speed_limit,
                photo_trigger_threshold=config.thresholds.photo_trigger_over_limit,
                continuous_detection=False,
                detection_interval=detection_interval,
            )

        # Start detection in separate thread
        detection_thread = threading.Thread(
            target=detection.run,
            args=(shutdown_event,),
            daemon=True,
        )
        detection_thread.start()
    else:
        logging.info("Detection module disabled in demo mode")

    # Initialize upload service
    logging.info("Starting upload service...")
    sync_service = SyncService(
        db=db,
        cloud_config=config.cloud,
        frequency_minutes=config.upload.frequency_minutes,
        batch_size=config.upload.batch_size,
        retry_attempts=config.upload.retry_attempts,
    )
    sync_service.start()

    # Create FastAPI app
    logging.info("Starting API server...")
    app = create_api(
        db=db,
        config=config,
        config_loader=config_loader,
        detection_module=detection,
        camera=camera if camera_ready else None,
        object_detector=object_detector if detector_ready else None,
        dashboard_overlays=dashboard_overlays,
        sync_service=sync_service,
    )

    # Start API server in separate thread
    api_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="warning",  # Only show warnings/errors, not every request
        access_log=False,  # Disable access logs for API requests
    )
    api_server = uvicorn.Server(api_config)

    api_thread = threading.Thread(target=api_server.run, daemon=True)
    api_thread.start()

    # Get local IP address for display
    local_ip = get_local_ip()

    logging.info("=" * 60)
    logging.info("RushRoster Field Device is running!")
    logging.info("Web dashboard:")
    logging.info(f"  Local:   http://localhost:8000")
    logging.info(f"  Network: http://{local_ip}:8000")
    if demo_mode:
        logging.info("  Mode: DEMO (hardware not connected)")
    logging.info("Press Ctrl+C to stop")
    logging.info("=" * 60)

    # Wait for shutdown signal
    try:
        shutdown_event.wait()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received")
        shutdown_event.set()

    # Graceful shutdown
    logging.info("Shutting down...")

    # Stop upload service
    sync_service.stop()

    # Stop API server
    api_server.should_exit = True

    # Close sensors
    if camera_ready:
        camera.close()
    if radar_ready:
        radar.close()

    # Wait for threads to finish
    if detection_thread and detection_thread.is_alive():
        detection_thread.join(timeout=2.0)

    logging.info("Shutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
