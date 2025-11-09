# RushRoster Field Device - Development Guide

This document provides guidance for AI assistants (like Claude) working on the RushRoster field device codebase.

## Project Overview

**RushRoster** is a Raspberry Pi-based speed monitoring system that uses FMCW radar for vehicle detection and USB cameras for photo capture. The field device:
- Detects passing vehicles and measures their speed
- Captures photos of speeding vehicles
- Stores events locally in SQLite
- Uploads data to a cloud platform for analysis

## Architecture

### Core Components

#### 1. **Sensors** (`src/sensors/`)
- **Radar Module** (`ops_radar_sensor.py`): Interfaces with PS243-C FMCW radar sensor via USB serial port
  - Returns real-time velocity measurements
  - Handles serial communication and frame parsing
  - Outputs speed in MPH

- **Camera Module** (`camera.py`): Manages continuous camera frame capture
  - Maintains a circular buffer of frames for burst capture
  - Triggered burst recording when speeding vehicle detected
  - Supports multiple camera devices

#### 2. **Detection** (`src/detection/`)
- **Burst Detection Module** (`burst_detection_module.py`): Core state machine for vehicle detection
  - Tracks vehicles over time with velocity filtering
  - Detects speeding (speed > speed_limit + photo_trigger_threshold)
  - Triggers burst photo capture when speeding detected
  - Creates event records in database
  - Includes frame selection logic to choose best photo from burst

- **Sync Module** (`sync_module.py`): Frame processor for burst capture
  - Selects best frame from burst sequence
  - Saves high-quality JPEG images
  - Updates event records with photo paths

#### 3. **Database** (`src/database/`)
- **SQLiteManager** (`sqlite_manager.py`): Local event storage
  - **Tables**:
    - `events`: Speed detections with photo references
    - `device_config`: Persistent configuration
    - `device_metadata`: Device location and settings
  - **Key columns**: `id`, `timestamp`, `speed`, `speed_limit`, `is_speeding`, `photo_path`, `uploaded`, `uploaded_at`
  - **Indexes**: `idx_events_timestamp`, `idx_events_uploaded`
  - Methods for querying, filtering, pagination, and statistics

#### 4. **API** (`src/api/`)
- **FastAPI Application** (`api.py`): Modular REST API with factory pattern
  - **Factory Function**: `create_api()` initializes FastAPI app with dependency injection
  - **Route Modules** (`routes/` subdirectory):
    - `status_routes.py`: `/health`, `/status` - Device health and system metrics
    - `stats_routes.py`: `/stats?hours=N`, `/events?limit=N&has_photos=BOOL` - Event statistics and filtering
    - `sensors_routes.py`: `/sensors/live`, `/sensors/stream` - Sensor snapshots and SSE streaming
    - `camera_routes.py`: `/camera/frame?detections=BOOL&hud=BOOL` - Live JPEG with optional overlays
    - `config_routes.py`: `/config`, `/config/cloud` - Device and cloud configuration management
    - `sync_routes.py`: `/upload`, `/download`, `/force-resync` - Cloud synchronization triggers
    - `ui_routes.py`: `/`, `/events/browse?page=N`, `/settings` - Web dashboard and UI pages
    - `events_routes.py`: `/events/delete`, `/events/clear`, `/events/photo/{id}` - Event management
  - **Data Models** (`models.py`): Pydantic models for validation (StatusResponse, StatsResponse, etc.)
  - **HUD Utility** (`hud.py`): Camera frame overlay rendering (speed, speed_limit, tracking state)

#### 5. **UI** (`src/ui/`)
- Web dashboard templates (Jinja2)
- HTML pages for diagnostics and configuration
- Event browser interface

#### 6. **Upload** (`src/upload/`)
- **Sync Service** (`sync_service.py`): Background service for cloud data synchronization
  - Fetches unuploaded events from database
  - Uploads events with photos to cloud API
  - Handles retries and error recovery
  - Marks events as uploaded in local database

#### 7. **Configuration** (`src/config.py`)
- Dataclass-based config management
- Sections: device, sensors, thresholds, upload, cloud, hud
- YAML loading and validation

## Key Algorithms & Flows

### Vehicle Detection Flow
1. Radar continuously outputs velocity measurements
2. Detection module applies threshold filter (speed >= min_speed)
3. Velocity measurements smoothed over time window
4. When sustained speed detected → vehicle tracking begins
5. If speed exceeds limit + threshold → burst photo capture triggered
6. Event recorded with initial `photo_path=None`
7. Async frame processing selects best frame
8. Event updated with photo path via `update_event_photo()`

### Photo Burst Capture
- Camera maintains circular buffer of recent frames (~30 frames for ~2 second buffer at 15fps)
- When speeding detected, burst recording begins
- Captures 20-30 frames over ~2 seconds
- Sync module evaluates frames (could use motion detection, face detection, clarity metrics)
- Selects best frame as representative photo
- Saves as JPEG to `photos/` directory

### Data Upload Pipeline
1. Sync service runs on scheduled interval (default 15 minutes)
2. Queries database for `uploaded=0` events (oldest first)
3. For each event with photo, uploads to cloud API
4. Cloud API returns success/failure
5. On success, marks event `uploaded=1` and sets `uploaded_at` timestamp
6. On failure, keeps `uploaded=0` for retry

## Database Schema

```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    speed REAL NOT NULL,
    speed_limit REAL NOT NULL,
    is_speeding BOOLEAN NOT NULL,
    photo_path TEXT,                    -- NULL if no photo captured
    uploaded BOOLEAN DEFAULT 0,
    uploaded_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)

-- Indexes for performance:
CREATE INDEX idx_events_timestamp ON events(timestamp)
CREATE INDEX idx_events_uploaded ON events(uploaded, timestamp)
```

## API Endpoints Reference

### Status Endpoints (`status_routes.py`)
- `GET /health` - Health check (returns status + timestamp)
- `GET /status` - Full device status with system metrics
  - Response: `{radar_status, camera_status, disk_usage_percent, disk_free_gb, upload_queue_size, network_connected, uptime_seconds}`
- `GET /favicon.ico` - Browser favicon (prevents 404)

### Statistics & Events (`stats_routes.py`)
- `GET /stats?hours=24` - Event statistics for time period
  - Response: `{total_events, speeding_events, avg_speed, max_speed, min_speed, period_hours}`
- `GET /events?limit=100&has_photos=true` - Recent events with optional photo filter
  - Query params: `limit` (default 100), `has_photos` (optional true/false)
  - Response: `{events: [...], count: N}`

### Sensor Data (`sensors_routes.py`)
- `GET /sensors/live` - Current sensor state snapshot
  - Response: `{radar: {current_speed, tracking, target_acquired}, camera: {active, frames_captured}, detection: {state, max_speed_this_target}}`
- `GET /sensors/stream` - Server-Sent Events (SSE) stream
  - Streams sensor data updates at ~200ms intervals (5 updates/sec)
  - Use with event listener: `new EventSource('/sensors/stream')`

### Camera & Live Feed (`camera_routes.py`)
- `GET /camera/frame?detections=true&hud=true` - Live JPEG frame
  - Query params: `detections` (overlay detection boxes), `hud` (show speed/speed_limit overlay)
  - Features: Auto-downscaling to 640px width, 1-second annotation cache, color-coded speed display
  - Response: JPEG image with Content-Type: image/jpeg

### Configuration (`config_routes.py`)
- `GET /config` - Current device and threshold configuration
  - Response includes: device info, location, speed_limit, thresholds, upload settings
- `PUT /config` - Update device settings
  - Request body: `{speed_limit?, photo_trigger_over_limit?, upload_frequency_minutes?}`
- `GET /config/cloud` - Cloud API configuration (API key masked for security)
- `PUT /config/cloud` - Update cloud credentials
  - Request body: `{api_url?, api_key?}`

### Cloud Sync (`sync_routes.py`)
- `POST /upload` - Trigger manual upload of unsynced events
  - Response: `{message, queue_size}`
- `POST /download` - Trigger download from cloud (background operation)
  - Response: `{message}`
- `POST /force-resync` - Mark all events as not-uploaded for retry
  - Response: `{message, events_reset}`

### Event Management (`events_routes.py`)
- `POST /events/delete` - Delete specific events by ID list
  - Request body: `{event_ids: [int, ...]}`
  - Response: `{deleted_count}`
- `POST /events/clear` - Delete all events (atomic operation)
  - Response: `{deleted_count}`
- `GET /events/photo/{event_id}` - Serve photo JPEG/PNG for event
  - Validates file existence and proper content-type
  - Supports absolute and relative photo paths

### Web UI (`ui_routes.py`)
- `GET /` - Main dashboard page (HTML)
  - Displays: status, 24-hour stats, device config, cloud config
- `GET /events/browse?page=1&has_photos=true` - Events browser with pagination
  - 20 events per page, includes total count and metadata
  - Optional photo filter
- `GET /settings` - Device settings configuration page (HTML)

## Configuration Management

### Device Settings (config.yaml)
```yaml
device:
  id: "rush-roster-001"
  location: {latitude, longitude, street_name}
  speed_limit: 25.0  # MPH

sensors:
  radar: {port: "/dev/ttyACM0"}
  camera: {device: 0, resolution: [1920, 1080]}

thresholds:
  speed_threshold_mph: 30.0
  photo_trigger_over_limit: 5.0
  min_speed: 10.0
  max_speed: 75.0

upload:
  frequency_minutes: 15
  batch_size: 100
  retry_attempts: 3

cloud:
  api_url: "https://api.speedmonitor.com"
  api_key: "***"

hud:
  enabled: true
  show_speed: true
  show_speed_limit: true
  show_tracking_state: true
```

### Loading & Updating
- `ConfigLoader.load()`: Reads YAML from disk
- `ConfigLoader.save()`: Persists changes to YAML
- Config objects are dataclasses (immutable by design—replace entire object on update)

## Common Development Tasks

### Adding a New API Endpoint
1. Choose the appropriate route module based on endpoint category (or create new module in `src/api/routes/`)
2. Define request/response Pydantic models in `src/api/models.py` if needed
3. Add `@app.get()` or `@app.post()` decorated function in the route module
4. Add documentation string with Args/Returns
5. Use `db.*` methods for data access
6. Handle exceptions with HTTPException
7. Register the route group in `create_api()` in `api.py` by calling `register_*()` function

Example (adding to `src/api/routes/events_routes.py`):
```python
# In src/api/routes/events_routes.py
@app.get("/events/search")
async def search_events(speed_min: float = 0, speed_max: float = 100):
    """Search events by speed range."""
    try:
        events = db.get_events_by_speed_range(speed_min, speed_max)
        return {"events": events, "count": len(events)}
    except Exception as e:
        logging.error(f"Error searching events: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def register_events_routes(app: FastAPI, db, config, sync_service):
    """Register event management endpoints."""
    app.include_router(router)  # router contains the endpoint
```

**Route Module Pattern:**
- Create a FastAPI `APIRouter` at module level
- Add all endpoints with decorator: `@router.get()`, `@router.post()`, etc.
- Define `register_*_routes()` function that registers router with app
- This function is called in `create_api()` with all required dependencies passed as arguments

### Adding Database Queries
1. Add method to `SQLiteManager` class in `sqlite_manager.py`
2. Use context manager pattern: `with self.get_connection() as conn:`
3. Use parameterized queries to prevent SQL injection: `cursor.execute(sql, params)`
4. Convert Row objects to dicts: `[dict(row) for row in cursor.fetchall()]`
5. Commit changes before closing connection

Example:
```python
def get_recent_events_filtered(self, limit: int = 100, has_photos: Optional[bool] = None):
    with self.get_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM events"
        params = []

        if has_photos is not None:
            query += " WHERE photo_path IS NOT NULL" if has_photos else " WHERE photo_path IS NULL"

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
```

### Adding Configuration Options
1. Add field to appropriate dataclass in `config.py`
2. Update `config.yaml.example`
3. Update `ConfigLoader.load()` if special parsing needed
4. Use in code via `config.<section>.<field>`

### Testing Detection Logic
- Use `test_burst_capture.py` as reference
- Can mock camera and radar inputs
- Focus on state machine transitions and event creation

## Important Patterns & Conventions

### Error Handling
- Use `logging.error()` for unexpected errors
- Use `logging.info()` for important operations
- Use `logging.debug()` for detailed diagnostics
- Always catch exceptions in API endpoints and return HTTPException

### Type Hints
- Use for all function parameters and returns
- Use `Optional[T]` for nullable values
- Use `List[Dict[str, Any]]` for event lists
- Use descriptive types over generic `Any` when possible

### Database Access
- Always use parameterized queries (not f-strings for SQL)
- Always use context managers for connections
- Always commit before exiting connection context
- Use `sqlite3.Row` with row_factory for column access by name

### API Response Format
- Wrap events in `{"events": [...], "count": N}` structure
- Use consistent snake_case for field names
- Include units in documentation (MPH, seconds, hours, etc.)
- Document optional query parameters

## File Locations & Paths

| Component | Location |
|-----------|----------|
| Config file | `/config.yaml` |
| SQLite database | `/data/events.db` |
| Photo storage | `/photos/` |
| Radar serial port | `/dev/ttyACM0` (configurable) |
| Camera device | `/dev/video0` (configurable) |
| Logs | `rush-roster.log` |
| Web templates | `src/ui/templates/` |

## Development Notes

### Known Limitations
- Frame selection in sync module currently uses basic heuristics
- No object detection for vehicle validation (can be added via YOLOv8)
- Photo upload to cloud is stubbed (awaiting cloud platform)
- No clustering of consecutive speeders (each vehicle = separate event)

### Performance Considerations
- Radar sensor updates ~10Hz, detection module samples ~5Hz
- Camera capture: continuous buffer reduces missed detections
- Frame processing is async to avoid blocking detection loop
- Database queries use indexes for fast filtering on timestamp/upload status
- API caches annotated frames (1 second TTL) to reduce CPU load on Raspberry Pi

### Future Enhancements
1. Object detection validation (confirm vehicles in photos)
2. License plate recognition (if required)
3. Clustering consecutive speeders from same vehicle
4. Geofencing (skip events outside specific areas)
5. Motion analysis for burst selection (select sharpest frame)
6. Advanced HUD rendering with vehicle tracking visualization

## Testing

### Unit Tests
- `test_burst_capture.py`: Tests detection module with simulated radar/camera
- Run with: `python test_burst_capture.py`

### Integration Testing
- Start application: `uv run main.py`
- Check API health: `curl http://localhost:8000/health`
- View dashboard: Open http://localhost:8000 in browser
- Trigger manual upload: `curl -X POST http://localhost:8000/upload`

### Hardware Testing
- Verify radar connection: `ls /dev/ttyACM*` and check serial communication
- Verify camera: `v4l2-ctl --list-devices` or test with OpenCV
- Monitor logs: `tail -f rush-roster.log`

## Git Workflow

- **Main branch**: Production-ready code
- **Commits**: Descriptive messages with intent (e.g., "Add has_photos filter to /events endpoint")
- **Recent commits**: Can view via `git log --oneline`

## Contact & Resources

- **Configuration Reference**: `config.yaml.example`
- **Main Entry**: `main.py`
