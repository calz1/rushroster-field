"""
Radar sensor interface for OPS24x FMCW and Doppler Radar Sensor.
Based on working implementation from original RushRoster project.
"""

import sys
import logging
import math
import serial
from typing import Optional


class RadarSensor:
    """Interface for OPS24x radar sensor with velocity measurement."""

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        speed_limit: float = 25.0,
        min_speed: float = 10.0,
        max_speed: float = 75.0,
        angle_degrees: float = 0.0,
        units: str = "US",  # US for MPH, UK for Km/H
        direction: str = "inbound",  # inbound, outbound, or bidirectional
    ):
        """
        Initialize radar sensor.

        Args:
            port: Serial port path (e.g., /dev/ttyACM0)
            speed_limit: Speed limit for the road (MPH)
            min_speed: Minimum speed to track (MPH)
            max_speed: Maximum speed to track (MPH)
            angle_degrees: Radar mounting angle for cosine adjustment
            units: Speed units (US=MPH, UK=Km/H)
            direction: Direction filter (inbound/outbound/bidirectional)
        """
        self.port = port
        self.speed_limit = speed_limit
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.angle_radians = math.radians(angle_degrees)
        self.cosine_adjustment = math.cos(self.angle_radians)
        self.units = units
        self.direction = direction

        self.serial_port: Optional[serial.Serial] = None
        self._is_initialized = False

        # Stationary vehicle filtering threshold (MPH)
        # Vehicles slower than this are considered stationary (parked, reflections)
        self.stationary_velocity_threshold = 3.0

        # OPS24x configuration parameters
        self._config = {
            "SAMPLING_FREQUENCY": "SX",  # 10Ksps
            "TRANSMIT_POWER": "PX",  # max power
            "MAGNITUDE_MIN": "M>40\n",  # Magnitude threshold (increased to filter parked vehicle reflections)
            "DECIMAL_DIGITS": "F0",  # No decimal reporting
            "BLANKS_PREF": "BZ",  # Send 0's not silence
            "LIVE_SPEED": "O1OS",  # Instantaneous speeds, single target
            "MAX_REPORTABLE": "R<200\n",
            "MIN_REPORTABLE": f"R>{int(min_speed)}\n",
            "UNITS_PREF": units,
        }

    def initialize(self) -> bool:
        """
        Initialize the radar sensor and configure it.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Open serial port
            self.serial_port = serial.Serial(
                port=self.port,
                baudrate=115200,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=0.1,
                write_timeout=2
            )

            self.serial_port.flushInput()
            self.serial_port.flushOutput()

            logging.info(f"Radar sensor opened on {self.port}")

            # Send configuration commands
            logging.info("Configuring OPS24x radar sensor...")
            self._send_command("Sampling Frequency", self._config["SAMPLING_FREQUENCY"])
            self._send_command("Transmit Power", self._config["TRANSMIT_POWER"])
            self._send_command("Magnitude Control", self._config["MAGNITUDE_MIN"])
            self._send_command("Decimal Digits", self._config["DECIMAL_DIGITS"])
            self._send_command("Min Speed", self._config["MIN_REPORTABLE"])
            self._send_command("Max Speed", self._config["MAX_REPORTABLE"])
            self._send_command("Units Preference", self._config["UNITS_PREF"])
            self._send_command("Blanks Preference", self._config["BLANKS_PREF"])
            self._send_command("Live Speed Mode", self._config["LIVE_SPEED"])

            # Set direction preference
            direction_cmd = self._get_direction_command()
            self._send_command("Direction Preference", direction_cmd)

            self._is_initialized = True
            logging.info("Radar sensor initialized successfully")
            return True

        except Exception as e:
            logging.error(f"Failed to initialize radar sensor: {e}")
            return False

    def _get_direction_command(self) -> str:
        """Get the direction command based on configuration."""
        if self.direction == "inbound":
            return "R+"
        elif self.direction == "outbound":
            return "R-"
        else:  # bidirectional
            return "R|"

    def _send_command(self, description: str, command: str) -> bool:
        """
        Send command to OPS24x module and verify response.

        Args:
            description: Human-readable description for logging
            command: Command string to send

        Returns:
            True if command acknowledged, False otherwise
        """
        if not self.serial_port:
            return False

        try:
            data_bytes = command.encode()
            logging.debug(f"Sending {description}: {command}")
            self.serial_port.write(data_bytes)

            # Wait for acknowledgment (starts with '{') with timeout
            verified = False
            max_attempts = 10  # Max 1 second (10 * 0.1s timeout)
            attempts = 0

            while not verified and attempts < max_attempts:
                response = self.serial_port.readline()
                attempts += 1

                if len(response) != 0:
                    response_str = str(response)
                    if '{' in response_str:
                        logging.debug(f"Response: {response_str}")
                        verified = True

            if not verified:
                logging.warning(f"No acknowledgment received for {description}")
                return False

            return True

        except Exception as e:
            logging.error(f"Error sending command {description}: {e}")
            return False

    def read_velocity(self) -> Optional[float]:
        """
        Read velocity from radar sensor.

        Returns:
            Velocity in MPH (positive=approaching, negative=moving away)
            None if no valid reading available
        """
        if not self.serial_port or not self._is_initialized:
            return None

        try:
            rx_bytes = self.serial_port.readline()

            if len(rx_bytes) == 0:
                return None

            rx_str = str(rx_bytes)

            # Check if this is a command response (contains '{')
            if '{' in rx_str:
                return None

            # Try to parse as velocity
            try:
                velocity = float(rx_bytes)

                # Apply angle adjustment: actual = measured / cos(angle)
                adjusted_velocity = velocity / self.cosine_adjustment

                return round(adjusted_velocity)

            except ValueError:
                return None

        except Exception as e:
            logging.error(f"Error reading velocity: {e}")
            return None

    def is_speed_valid(self, velocity: Optional[float]) -> bool:
        """
        Check if velocity is within valid tracking range.

        Args:
            velocity: Velocity to check (can be None)

        Returns:
            True if velocity is within min/max bounds
        """
        if velocity is None:
            return False

        abs_velocity = abs(velocity)

        # CRITICAL: Filter out stationary targets (parked vehicles, clutter, reflections)
        # This prevents false positives from parked cars on shoulder lanes
        if abs_velocity < self.stationary_velocity_threshold:
            return False

        return self.min_speed < abs_velocity < self.max_speed

    def is_speeding(self, velocity: Optional[float]) -> bool:
        """
        Check if velocity exceeds speed limit.

        Args:
            velocity: Velocity to check (can be None)

        Returns:
            True if velocity exceeds speed limit
        """
        if velocity is None:
            return False
        return abs(velocity) > self.speed_limit

    def close(self):
        """Close the serial port and release resources."""
        if self.serial_port:
            self.serial_port.close()
            self._is_initialized = False
            logging.info("Radar sensor closed")

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
