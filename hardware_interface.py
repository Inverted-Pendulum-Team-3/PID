#!/usr/bin/env python3
"""
Hardware interface for ML agent to read sensor data in real-time.
Uses low-latency observation cache file when fresh (sensors write each cycle),
otherwise falls back to SQLite. Reduces end-to-end delay for deploy.
"""

import os
import struct
import sqlite3
import time
import numpy as np
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(_SCRIPT_DIR, "sensor_data.db")
OBS_CACHE_FILE = os.path.join(_SCRIPT_DIR, "obs_cache.bin")
OBS_CACHE_MAX_AGE_S = 0.05   # use cache only if timestamp within 50 ms


def get_sensor_data(target_velocity=0.0, target_rotation_rate=0.0):
    """
    Get latest observation vector. Tries low-latency cache first, then SQLite.
    Computes velocity_error and rotation_error from the two control inputs.

    Args:
        target_velocity: Desired forward speed (m/s). Error = target_velocity - actual.
        target_rotation_rate: Desired turning rate (rad/s). Error = target_rotation_rate - actual.

    Returns [linear_velocity, pitch, pitch_rate, yaw_rate,
             wheel_velocity_1, wheel_velocity_2, velocity_error, rotation_error, 0].
    Returns None if no data.

    Order:
      [0] linear_velocity (m/s)
      [1] pitch (rad), [2] pitch_rate (rad/s), [3] yaw_rate (rad/s)
      [4] wheel_velocity_1, [5] wheel_velocity_2
      [6] velocity_error   = target_velocity - linear_velocity
      [7] rotation_error  = target_rotation_rate - yaw_rate
      [8] reserved (0).
    """
    def _apply_errors(obs):
        obs = np.asarray(obs, dtype=np.float32)
        if len(obs) >= 8:
            obs[6] = float(target_velocity) - float(obs[0])
            obs[7] = float(target_rotation_rate) - float(obs[3])
        return obs

    # Prefer cache file (written every sensor cycle) for minimal delay
    try:
        if os.path.isfile(OBS_CACHE_FILE):
            with open(OBS_CACHE_FILE, "rb") as f:
                raw = f.read()
            if len(raw) >= 8 + 9 * 4:
                ts, = struct.unpack("<d", raw[:8])
                obs = list(struct.unpack("<9f", raw[8:8 + 36]))
                if (time.monotonic() - ts) <= OBS_CACHE_MAX_AGE_S:
                    return _apply_errors(obs)
    except Exception:
        pass

    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM sensor_readings
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None

        linear_velocity = row["robot_v"] or 0.0
        pitch = row["imu1_body_pitch"] or 0.0   # angle of tilt forwards/backwards (rad)
        pitch_rate = row["imu1_gy"] or 0.0      # rate of tilt forwards/backwards (rad/s), gyro Y
        # Yaw rate now comes from IMU yaw_rate column (gyro Z) instead of encoder-based robot_w_enc
        yaw_rate = row["imu1_yaw_rate"] or 0.0
        wheel_velocity_1 = row["encoder_left_rad_s"] or 0.0
        wheel_velocity_2 = row["encoder_right_rad_s"] or 0.0
        velocity_error = target_velocity - linear_velocity
        rotation_error = target_rotation_rate - yaw_rate
        
        return np.array([
            linear_velocity,
            pitch,
            pitch_rate,
            yaw_rate,
            wheel_velocity_1,
            wheel_velocity_2,
            velocity_error,
            rotation_error,
            0,
        ], dtype=np.float32)
        
    except Exception as e:
        print(f"[hardware_interface] Error reading sensor data: {e}")
        return None


# Single shared pigpio connection (avoid "too many connections" at 100 Hz)
_pi = None
_pigpio_warned = False

GPIO_M1 = 9   # S1 — Motor 1
GPIO_M2 = 25  # S2 — Motor 2


def _get_pi():
    """Get or create the single pigpio connection. Returns None if unavailable."""
    global _pi, _pigpio_warned
    try:
        import pigpio
    except ImportError:
        if not _pigpio_warned:
            print("[hardware_interface] pigpio not installed. Motor commands disabled.")
            _pigpio_warned = True
        return None
    if _pi is not None and _pi.connected:
        return _pi
    if _pi is not None:
        try:
            _pi.stop()
        except Exception:
            pass
        _pi = None
    _pi = pigpio.pi()
    if not _pi.connected:
        if not _pigpio_warned:
            print("[hardware_interface] pigpio not connected. Run: sudo pigpiod")
            _pigpio_warned = True
        return None
    return _pi


def set_motor_velocities(left_action, right_action):
    """
    Send motor commands to Sabertooth 2x12.
    Uses a single shared pigpio connection (safe for 100 Hz loop).

    Args:
        left_action: Action value for left motor (-1 to 1)
        right_action: Action value for right motor (-1 to 1)
    """
    pi = _get_pi()
    if pi is None:
        return False
    try:
        pwm_left = int(1500 + (left_action * 500))
        pwm_right = int(1500 + (right_action * 500))
        pwm_left = max(1000, min(2000, pwm_left))
        pwm_right = max(1000, min(2000, pwm_right))
        pi.set_servo_pulsewidth(GPIO_M1, pwm_left)
        pi.set_servo_pulsewidth(GPIO_M2, pwm_right)
        return True
    except Exception as e:
        print(f"[hardware_interface] Error setting motor velocities: {e}")
        return False


def close_motor_connection():
    """Stop motors and close the pigpio connection. Call on exit (e.g. deploy finally)."""
    global _pi, _pigpio_warned
    if _pi is not None:
        try:
            # Send explicit stop (1500 µs) so Sabertooth stops before we disconnect; 0 = no signal.
            _pi.set_servo_pulsewidth(GPIO_M1, 1500)
            _pi.set_servo_pulsewidth(GPIO_M2, 1500)
            time.sleep(0.05)
            _pi.set_servo_pulsewidth(GPIO_M1, 0)
            _pi.set_servo_pulsewidth(GPIO_M2, 0)
            _pi.stop()
        except Exception:
            pass
        _pi = None
