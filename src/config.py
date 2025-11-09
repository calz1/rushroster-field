"""
Configuration management for field device.
Loads settings from config.yaml file.
"""

import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class DeviceConfig:
    """Device identification and location."""

    id: str = "unknown-device"
    latitude: float = 0.0
    longitude: float = 0.0
    street_name: str = "Unknown Street"
    speed_limit: float = 25.0


@dataclass
class RadarConfig:
    """Radar sensor configuration."""

    port: str = "/dev/ttyACM0"
    sample_rate: int = 10


@dataclass
class CameraConfig:
    """Camera configuration."""

    device: int = 0
    resolution: list[int] = field(default_factory=lambda: [1920, 1080])
    target_name: Optional[str] = None
    zoom_factor: float = 1.0


@dataclass
class ThresholdConfig:
    """Detection thresholds."""

    speed_threshold_mph: float = 30.0
    photo_trigger_over_limit: float = 5.0
    min_speed: float = 10.0
    max_speed: float = 75.0


@dataclass
class UploadConfig:
    """Upload service configuration."""

    frequency_minutes: int = 15
    batch_size: int = 100
    retry_attempts: int = 3


@dataclass
class CloudConfig:
    """Cloud platform connection."""

    api_url: str = "https://api.speedmonitor.com"
    api_key: str = ""


@dataclass
class HUDConfig:
    """Camera HUD overlay configuration."""

    enabled: bool = True
    show_speed: bool = True
    show_speed_limit: bool = True
    show_tracking_state: bool = True


@dataclass
class LogRotationConfig:
    """Log rotation configuration."""

    max_bytes: int = 10485760  # 10MB default
    backup_count: int = 5  # Keep 5 backup files


@dataclass
class LoggingConfig:
    """Logging configuration."""

    enabled: bool = True
    log_dir: str = "logs"
    rotation: LogRotationConfig = field(default_factory=LogRotationConfig)


@dataclass
class Config:
    """Complete field device configuration."""

    device: DeviceConfig = field(default_factory=DeviceConfig)
    sensors: Dict[str, Any] = field(default_factory=dict)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    hud: HUDConfig = field(default_factory=HUDConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


class ConfigLoader:
    """Loads and manages configuration from YAML file."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize config loader.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config: Optional[Config] = None

    def load(self) -> Config:
        """
        Load configuration from file.

        Returns:
            Loaded configuration object

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        if not self.config_path.exists():
            logging.warning(
                f"Config file not found: {self.config_path}, using defaults"
            )
            self.config = Config()
            return self.config

        try:
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f)

            if data is None:
                data = {}

            # Parse device config
            device_data = data.get("device", {})
            device = DeviceConfig(
                id=device_data.get("id", "unknown-device"),
                latitude=device_data.get("location", {}).get("latitude", 0.0),
                longitude=device_data.get("location", {}).get("longitude", 0.0),
                street_name=device_data.get("location", {}).get(
                    "street_name", "Unknown Street"
                ),
                speed_limit=device_data.get("speed_limit", 25.0),
            )

            # Parse sensor configs
            sensors_data = data.get("sensors", {})
            radar_data = sensors_data.get("radar", {})
            camera_data = sensors_data.get("camera", {})

            radar = RadarConfig(
                port=radar_data.get("port", "/dev/ttyACM0"),
                sample_rate=radar_data.get("sample_rate", 10),
            )

            camera = CameraConfig(
                device=camera_data.get("device", 0),
                resolution=camera_data.get("resolution", [1920, 1080]),
                target_name=camera_data.get("target_name"),
                zoom_factor=camera_data.get("zoom_factor", 1.0),
            )

            # Parse thresholds
            thresholds_data = data.get("thresholds", {})
            thresholds = ThresholdConfig(
                speed_threshold_mph=thresholds_data.get("speed_threshold_mph", 30.0),
                photo_trigger_over_limit=thresholds_data.get(
                    "photo_trigger_over_limit", 5.0
                ),
                min_speed=thresholds_data.get("min_speed", 10.0),
                max_speed=thresholds_data.get("max_speed", 75.0),
            )

            # Parse upload config
            upload_data = data.get("upload", {})
            upload = UploadConfig(
                frequency_minutes=upload_data.get("frequency_minutes", 15),
                batch_size=upload_data.get("batch_size", 100),
                retry_attempts=upload_data.get("retry_attempts", 3),
            )

            # Parse cloud config
            cloud_data = data.get("cloud", {})
            cloud = CloudConfig(
                api_url=cloud_data.get("api_url", "https://api.speedmonitor.com"),
                api_key=cloud_data.get("api_key", ""),
            )

            # Parse HUD config
            hud_data = data.get("hud", {})
            hud = HUDConfig(
                enabled=hud_data.get("enabled", True),
                show_speed=hud_data.get("show_speed", True),
                show_speed_limit=hud_data.get("show_speed_limit", True),
                show_tracking_state=hud_data.get("show_tracking_state", True),
            )

            # Parse logging config
            logging_data = data.get("logging", {})
            log_rotation = LogRotationConfig(
                max_bytes=logging_data.get("max_bytes", 10485760),
                backup_count=logging_data.get("backup_count", 5),
            )
            logging_config = LoggingConfig(
                enabled=logging_data.get("enabled", True),
                log_dir=logging_data.get("log_dir", "logs"),
                rotation=log_rotation,
            )

            # Keep raw object_detector config for use in main.py
            object_detector_data = sensors_data.get("object_detector", {})

            self.config = Config(
                device=device,
                sensors={"radar": radar, "camera": camera, "object_detector": object_detector_data},
                thresholds=thresholds,
                upload=upload,
                cloud=cloud,
                hud=hud,
                logging=logging_config,
            )

            logging.info(f"Configuration loaded from {self.config_path}")
            return self.config

        except yaml.YAMLError as e:
            logging.error(f"Error parsing config file: {e}")
            raise

        except Exception as e:
            logging.error(f"Error loading config: {e}")
            raise

    def save(self, config: Config):
        """
        Save configuration to file.

        Args:
            config: Configuration object to save
        """
        try:
            # Convert config to dict
            radar_cfg = config.sensors.get("radar", RadarConfig())
            camera_cfg = config.sensors.get("camera", CameraConfig())

            data = {
                "device": {
                    "id": config.device.id,
                    "location": {
                        "latitude": config.device.latitude,
                        "longitude": config.device.longitude,
                        "street_name": config.device.street_name,
                    },
                    "speed_limit": config.device.speed_limit,
                },
                "sensors": {
                    "radar": {
                        "port": radar_cfg.port,
                        "sample_rate": radar_cfg.sample_rate,
                    },
                    "camera": {
                        "device": camera_cfg.device,
                        "resolution": camera_cfg.resolution,
                        "target_name": camera_cfg.target_name,
                        "zoom_factor": camera_cfg.zoom_factor,
                    },
                },
                "thresholds": {
                    "speed_threshold_mph": config.thresholds.speed_threshold_mph,
                    "photo_trigger_over_limit": config.thresholds.photo_trigger_over_limit,
                    "min_speed": config.thresholds.min_speed,
                    "max_speed": config.thresholds.max_speed,
                },
                "upload": {
                    "frequency_minutes": config.upload.frequency_minutes,
                    "batch_size": config.upload.batch_size,
                    "retry_attempts": config.upload.retry_attempts,
                },
                "cloud": {
                    "api_url": config.cloud.api_url,
                    "api_key": config.cloud.api_key,
                },
                "hud": {
                    "enabled": config.hud.enabled,
                    "show_speed": config.hud.show_speed,
                    "show_speed_limit": config.hud.show_speed_limit,
                    "show_tracking_state": config.hud.show_tracking_state,
                },
                "logging": {
                    "enabled": config.logging.enabled,
                    "log_dir": config.logging.log_dir,
                    "max_bytes": config.logging.rotation.max_bytes,
                    "backup_count": config.logging.rotation.backup_count,
                },
            }

            with open(self.config_path, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

            logging.info(f"Configuration saved to {self.config_path}")

        except Exception as e:
            logging.error(f"Error saving config: {e}")
            raise


def create_default_config(path: str = "config.yaml"):
    """
    Create a default configuration file.

    Args:
        path: Path where to create the config file
    """
    config = Config()
    loader = ConfigLoader(path)
    loader.save(config)
    logging.info(f"Default configuration created at {path}")
