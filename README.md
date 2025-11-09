# RushRoster Field Device

## The Origin Story

My neighborhood has a dangerous road with no crosswalk, despite kids needing to cross it daily to get to school. Local authorities were skeptical about the speeding problem, so I decided to build a system to collect data and prove it. The result is RushRoster: a full-featured platform that demonstrates how multiple sensors and AI can work together without a lot of exotic hardware.

This software for the "field device" is the core of the RushRoster speed monitoring system. It runs on a Raspberry Pi with a radar sensor and camera attached to detect and photograph speeding vehicles. This software works standalone or can optionally connect to RushRoster Cloud for centralized data management across multiple devices.

## Features

- 🎯 **Real-time Speed Detection**: Radar sensor integration for speed measurement
- 📸 **AI-Powered Photo Selection**: AI analyzes a series of photos and selects the best one for clear, frame-optimized evidence
- 🔋 **Standalone Operation**: Functions independently without cloud connectivity
- 🌐 **REST API**: Complete diagnostic and control API for device management
- 📱 **Web Dashboard**: Real-time status, metrics, and configuration management
- ☁️ **Optional Cloud Integration**: Background sync service for multi-device centralized management

## Technology Stack

- **Runtime**: Python 3.11+ with UV package manager
- **Web Framework**: FastAPI + Uvicorn
- **Database**: SQLite3 (local)
- **Hardware Interface**: PySerial (radar), OpenCV (camera), Supervision (object detection)
- **Frontend**: HTML/CSS/JavaScript with server-sent events (SSE) for real-time updates
- **Config**: YAML-based configuration management

## Hardware Requirements

- **Raspberry Pi 4**: running Raspbian Linux (Bookworm or later)
- **Network connectivity**: Ethernet or WiFi - I use Power over Ethernet (PoE) with RJ-45 + USB C to power the Pi too. 
- **Sensors**
    - **OPS243-A Radar Sensor**: This software expects one of these plugged in via USB - https://omnipresense.com/product/ops243-doppler-radar-sensor/
    - **USB Webcam**: most any USB webcam supported on Linux will work. I originally started with a Logitech C920, but the FOV was too wide. I have switched to this which is higher resolution and has a zoom lens - https://www.aliexpress.us/item/3256808555869000.html 
- **SD card**: At least 16GB storage. Depending on how long it runs, how many photos it takes, and how often you upload data to the central server, you may want more.

## Software Components

- **Radar Sensor Module**: Polls the radar for speed detection
- **Camera Module**: synchronized photo capture with YOLOv4-tiny vehicle detection
- **Detection Module**: State machine for vehicle tracking and event recording with intelligent frame filtering
- **SQLite Database**: Local storage for events and metadata
- **FastAPI**: REST API for device status and configuration
- **Web Dashboard**: Simple interface for viewing the cameras and latest events
- **Sync Service**: Background service for uploading data to cloud 



## Quick Start

```bash
# 1. Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and navigate
git clone https://github.com/calz1/rushroster-field.git
cd rush-roster/field

# 3. Install dependencies
uv sync

# 4. Configure
cp config.yaml.example config.yaml
# Edit config.yaml with your device settings

# 5. Run
uv run main.py

# 6. Access dashboard
# Open http://<RPI IP>:8000 in your browser
```

## Dashboard

Access the web dashboard at **http://<RPI IP>:8000**

**Real-time Monitoring:**
- Live camera feed with HUD overlay
- Current speed readings (5 updates/sec via SSE)
- Device health indicators (radar, camera, network)
- Storage usage and upload queue status
- 24-hour statistics (total vehicles, speeders, average/max speed)
- Configuration viewer and quick actions

### Dashboard Preview

![Dashboard Screenshot](example_media/example.png)
*RushRoster Field Device Dashboard with live speed readings, camera feed, and system statistics*

### Video Demo

<video src="https://github.com/user-attachments/assets/b387428d-2e24-4c93-937e-480cb6decec9" width="581" height="618"></video>

### Lightweight System Performance

![System Resource Usage](example_media/howbusy.png)
*Here's what the system looks like with RushRoster running*

## API Endpoints

- `GET /health` - Health check
- `GET /status` - Device status and sensor info
- `GET /stats` - Statistics (vehicles, speeders, speeds)
- `GET /events` - Recent detected events
- `GET /config` - Current device configuration
- `PUT /config` - Update configuration
- `POST /sync` - Trigger manual cloud sync
- `GET /camera/frame` - Current camera frame with HUD
- `GET /sensors/stream` - Real-time sensor data (Server-Sent Events)

Full API docs available at `/docs` (Swagger) and `/redoc` (ReDoc)
## Logging

Logs are written to console and `rush-roster.log`. Adjust log level in `main.py` if needed.

## Troubleshooting

### Camera not found
```bash
ls /dev/video*  # Check available cameras
```
Update `camera.device` in config.yaml to match the correct device path.

### Radar not detected
```bash
ls /dev/ttyACM*  # Check USB serial devices
```
Update `radar.port` in config.yaml. Also ensure user has serial permissions:
```bash
sudo usermod -a -G dialout $USER
# Log out and back in for changes to take effect
```

