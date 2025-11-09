"""Detection modules for synchronized radar and camera tracking."""

from .sync_module import SyncDetectionModule, DetectionEvent
from .burst_detection_module import BurstDetectionModule

__all__ = ["SyncDetectionModule", "DetectionEvent", "BurstDetectionModule"]
