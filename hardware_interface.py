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
import queue as _queue
import threading as _threading
import numpy as np
from multiprocessing import shared_memory as _shm_mod

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(_SCRIPT_DIR, "sensor_data.db")
OBS_CACHE_FILE = os.path.join(_SCRIPT_DIR, "obs_cache.bin")
OBS_CACHE_MAX_AGE_S = 0.05   # use cache only if timestamp within 50 ms

# ---------------------------------------------------------------------------
# Background motor-state writer — keeps SD card I/O off the PID hot loop.
# set_motor_velocities() drops the latest state into this queue (non-blocking).
# The writer thread drains it and does the atomic tmp+replace write.
# ---------------------------------------------------------------------------
_motor_state_queue: _queue.Queue = _queue.Queue(maxsize=2)


def _motor_state_writer() -> None:
    _mf  = os.path.join(_SCRIPT_DIR, "motor_state.bin")
    _tmp = _mf + ".tmp"
    while True:
        try:
            buf = _motor_state_queue.get(timeout=1.0)
            with open(_tmp, "wb") as f:
                f.write(buf)
            os.replace(_tmp, _mf)
        except _queue.Empty:
            pass
        except Exception:
            pass


_threading.Thread(target=_motor_state_writer, daemon=True,
                  name="motor-state-writer").start()


SHM_NAME = "sensors_shm"
SHM_SIZE = 8 + 9 * 4  # must match sensorsHI.py

_shm_reader = None  # module-level cached handle

# If SHM data is older than this it belongs to a dead/restarted sensor session.
# Close the cached handle so the next call reopens and binds to the new SHM.
_SHM_STALE_RECONNECT_S = 2.0


def _get_shm_reader():
    """Lazily open shared memory. Returns the SharedMemory object or None."""
    global _shm_reader
    if _shm_reader is not None:
        try:
            _ = _shm_reader.buf  # check still alive
            return _shm_reader
        except Exception:
            _shm_reader = None
    try:
        _shm_reader = _shm_mod.SharedMemory(name=SHM_NAME, create=False, size=SHM_SIZE)
        return _shm_reader
    except Exception:
        return None


def _close_shm_reader():
    """Invalidate the cached SHM handle so the next call reopens to the current SHM."""
    global _shm_reader
    if _shm_reader is not None:
        try:
            _shm_reader.close()
        except Exception:
            pass
        _shm_reader = None


def get_shm_age_ms():
    """
    Return the age of the data currently in shared memory, in milliseconds.
    This reflects how long ago sensors_3_28.py last wrote a fresh observation.
    Returns None if SHM is unavailable, not yet written (ts == 0), or stale
    from a previous sensor session (age > _SHM_STALE_RECONNECT_S).
    """
    shm = _get_shm_reader()
    if shm is None:
        return None
    try:
        ts, = struct.unpack_from("<d", shm.buf, 0)
        if ts == 0.0:
            return None
        age_s = time.monotonic() - ts
        if age_s > _SHM_STALE_RECONNECT_S:
            # Handle points to old SHM from a previous session — reopen next call.
            _close_shm_reader()
            return None
        return age_s * 1000.0
    except Exception:
        return None


def get_sensor_data(target_velocity=0.0, target_rotation_rate=0.0):
    """
    Get latest observation vector.
    Priority: shared memory (zero-copy) → file cache → SQLite.
    """
    def _apply_errors(obs):
        obs = np.asarray(obs, dtype=np.float32)
        if len(obs) >= 8:
            obs[6] = float(target_velocity) - float(obs[0])
            obs[7] = float(target_rotation_rate) - float(obs[3])
        return obs

    # 1. Shared memory — lowest latency, no allocation
    shm = _get_shm_reader()
    if shm is not None:
        try:
            buf = shm.buf                              # memoryview — no copy
            ts, = struct.unpack_from("<d", buf, 0)
            age = time.monotonic() - ts
            if age > _SHM_STALE_RECONNECT_S:
                # Stale handle from a previous sensor session — reopen next call.
                _close_shm_reader()
            elif age <= OBS_CACHE_MAX_AGE_S:
                obs = list(struct.unpack_from("<9f", buf, 8))
                return _apply_errors(obs)
        except Exception:
            pass

    # 2. File cache fallback
    try:
        if os.path.isfile(OBS_CACHE_FILE):
            with open(OBS_CACHE_FILE, "rb") as f:
                raw = f.read(SHM_SIZE)               # only read what we need
            if len(raw) >= SHM_SIZE:
                ts, = struct.unpack_from("<d", raw, 0)
                obs = list(struct.unpack_from("<9f", raw, 8))
                if (time.monotonic() - ts) <= OBS_CACHE_MAX_AGE_S:
                    return _apply_errors(obs)
    except Exception:
        pass

    # 3. SQLite last resort
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sensor_readings ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        if row is None:
            return None
        linear_velocity  = row["robot_v"] or 0.0
        pitch            = row["imu1_body_pitch"] or 0.0
        pitch_rate       = row["imu1_gy"] or 0.0
        yaw_rate         = row["imu1_yaw_rate"] or 0.0
        wheel_velocity_1 = row["encoder_left_rad_s"] or 0.0
        wheel_velocity_2 = row["encoder_right_rad_s"] or 0.0
        return _apply_errors([
            linear_velocity, pitch, pitch_rate, yaw_rate,
            wheel_velocity_1, wheel_velocity_2, 0.0, 0.0, 0.0,
        ])
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
    # Send neutral immediately on first connection so Sabertooth sees a valid
    # stop signal before any control loop runs — prevents random motor spin on power-on
    _pi.set_servo_pulsewidth(GPIO_M1, 1500)
    _pi.set_servo_pulsewidth(GPIO_M2, 1500)
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

    # Guard against NaN/inf from upstream (bad sensor reading, division error, etc.)
    import math as _math
    if not _math.isfinite(left_action) or not _math.isfinite(right_action):
        print(f"[hardware_interface] Non-finite motor cmd dropped: L={left_action} R={right_action}")
        return False

    pwm_left  = max(1000, min(2000, int(1500 + left_action  * 500)))
    pwm_right = max(1000, min(2000, int(1500 + right_action * 500)))

    ok = True
    # Send each channel independently — one pigpio error must not silence the other motor.
    try:
        pi.set_servo_pulsewidth(GPIO_M1, pwm_left)
    except Exception as e:
        print(f"[hardware_interface] M1 (GPIO {GPIO_M1}) error: {e}")
        ok = False
    try:
        pi.set_servo_pulsewidth(GPIO_M2, pwm_right)
    except Exception as e:
        print(f"[hardware_interface] M2 (GPIO {GPIO_M2}) error: {e}")
        ok = False

    # Queue motor state for webserver display — never blocks the PID loop.
    try:
        _buf = struct.pack("<dff", time.monotonic(), float(left_action), float(right_action))
        _motor_state_queue.put_nowait(_buf)
    except _queue.Full:
        pass  # webserver display lags by one extra frame — acceptable
    return ok


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

