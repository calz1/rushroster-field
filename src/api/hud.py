"""HUD (Head-Up Display) rendering for camera frames."""

from typing import Any, Dict


def render_hud_overlay(frame, speed: float | None, speed_limit: float, tracking: bool, hud_config: Dict[str, bool]) -> Any:
    """
    Render HUD overlay on camera frame.

    Args:
        frame: Camera frame (numpy array, BGR format)
        speed: Current speed in MPH (None if not available)
        speed_limit: Speed limit in MPH
        tracking: Whether radar is currently tracking
        hud_config: HUD configuration dictionary with display flags

    Returns:
        Frame with HUD overlay
    """
    import cv2

    # Make a copy to avoid modifying original
    frame = frame.copy()
    height, width = frame.shape[:2]

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    # Colors (BGR format)
    white = (255, 255, 255)
    black = (0, 0, 0)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    red = (0, 0, 255)
    gray = (128, 128, 128)

    # Background color and transparency
    overlay = frame.copy()

    y_offset = 30
    x_offset = 20
    line_height = 40

    # Show speed if configured
    if hud_config.get('show_speed', True) and speed is not None:
        # Determine color based on speed
        if speed > 35:
            speed_color = red
        elif speed > 25:
            speed_color = yellow
        else:
            speed_color = green

        speed_text = f"Speed: {speed:.0f} MPH"
        # Draw semi-transparent background
        text_size = cv2.getTextSize(speed_text, font, font_scale * 1.5, thickness)[0]
        cv2.rectangle(overlay,
                     (x_offset - 5, y_offset - text_size[1] - 5),
                     (x_offset + text_size[0] + 5, y_offset + 5),
                     black, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        overlay = frame.copy()

        # Draw speed text with color coding
        cv2.putText(frame, speed_text, (x_offset, y_offset), font,
                   font_scale * 1.5, speed_color, thickness)
        y_offset += line_height

    # Show speed limit if configured
    if hud_config.get('show_speed_limit', True):
        limit_text = f"Limit: {speed_limit:.0f} MPH"
        text_size = cv2.getTextSize(limit_text, font, font_scale, thickness)[0]
        cv2.rectangle(overlay,
                     (x_offset - 5, y_offset - text_size[1] - 5),
                     (x_offset + text_size[0] + 5, y_offset + 5),
                     black, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        overlay = frame.copy()

        cv2.putText(frame, limit_text, (x_offset, y_offset), font,
                   font_scale, white, thickness)
        y_offset += line_height

    # Show tracking state if configured
    if hud_config.get('show_tracking_state', True):
        tracking_text = "Tracking" if tracking else "Idle"
        tracking_color = green if tracking else gray
        text_size = cv2.getTextSize(tracking_text, font, font_scale, thickness)[0]
        cv2.rectangle(overlay,
                     (x_offset - 5, y_offset - text_size[1] - 5),
                     (x_offset + text_size[0] + 5, y_offset + 5),
                     black, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        overlay = frame.copy()

        cv2.putText(frame, tracking_text, (x_offset, y_offset), font,
                   font_scale, tracking_color, thickness)

    return frame
