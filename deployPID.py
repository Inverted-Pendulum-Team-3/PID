#!/usr/bin/env python3
"""
PIDLinear.py — Balance PID with smooth power curve + deadband compensation.

Same as PID.py but with two extra features for better small-angle performance:
  1. Smooth linear power ramp (TILT_PWR_MIN → 1.0) instead of a step function,
     giving the D term more authority at small tilt angles.
  2. Motor deadband compensation — rescales commands so any non-trivial PID
     output jumps over the Sabertooth's ~±30 µs dead zone.

Per-wheel velocity equalisation is also included (disabled by default,
set WHEEL_EQ_GAIN > 0 to enable).
"""

import os
import sys
import math
import time
import struct
import signal
import json as _json
import threading as _threading
from collections import deque
from datetime import datetime
from multiprocessing import shared_memory as _shm_mod
from multiprocessing import resource_tracker as _res_tracker

import numpy as np

from hardware_interface import (
    set_motor_velocities,
    close_motor_connection,
)

# Shared memory — must match sensors_3_28.py
_SHM_NAME = "sensors_shm"
_SHM_SIZE = 8 + 9 * 4   # float64 ts + 9 × float32

# Cut motors if SHM is older than this.
# sensors_3_28 writes SHM every ~10 ms (100 Hz); > 150 ms = 15 missed loops.
IMU_STALE_LIMIT_S = 1.0

_shm_handle = None


def _read_shm():
    """
    Read obs vector directly from shared memory.
    Returns (obs_list, age_s) on success, or (None, 9999.0) on any failure.
    Never touches the filesystem or SQLite — zero I/O latency.
    """
    global _shm_handle
    if _shm_handle is None:
        try:
            _shm_handle = _shm_mod.SharedMemory(name=_SHM_NAME, create=False, size=_SHM_SIZE)
        except Exception:
            return None, 9999.0
        # We're a consumer, not the owner — prevent this process's
        # resource_tracker from unlinking the SHM segment on exit.
        try:
            _res_tracker.unregister(_shm_handle._name, "shared_memory")
        except Exception:
            pass
    try:
        buf  = _shm_handle.buf                   # memoryview — no allocation
        ts,  = struct.unpack_from("<d", buf, 0)
        if ts == 0.0:
            _shm_handle.close()
            _shm_handle = None
            return None, 9999.0
        age  = time.monotonic() - ts
        if age > IMU_STALE_LIMIT_S:
            _shm_handle.close()
            _shm_handle = None
            return None, 9999.0
        obs  = list(struct.unpack_from("<9f", buf, 8))
        return obs, age
    except Exception:
        _shm_handle = None
        return None, 9999.0

# ---------------------------------------------------------------------------
# PID / control gains
# ---------------------------------------------------------------------------
KP_PITCH      = 43.5
KD_PITCH      = 9.0
KI_PITCH      =  0.0
I_PITCH_MAX   =  2.0
WHEEL_CMD_MAX = 12.8
K_WHEEL_AVG   =  0.0

# Low-pass filter coefficient for gyro derivative term (0–1).
# Lower = more smoothing but more lag.  0.1–0.2 is a good starting point.
GYRO_LPF_ALPHA = 0.15

FWD_MAX   =  2.0
FWD_ACCEL = 10.0
FWD_DECAY =  0.5

YAW_MAX   =  2.0
YAW_ACCEL = 10.0

K_YAW_HOLD  =  1.0
K_YAW_RATE  =  0.2
YAW_CMD_MAX =  1.0

# Maximum yaw correction as a fraction of the current balance command.
# Prevents one motor from running at nearly double the other during
# heading hold when the balance command is small.
# 0.25 → worst-case L/R ratio ≈ 1.67:1.  Set to 1.0 to disable.
MAX_YAW_FRAC = 0.25
# Floor so the robot can still make small heading corrections when nearly balanced.
MIN_YAW_ABS  = 0.15

PITCH_TIP_LIMIT = math.radians(35)

# Smooth power curve: linear ramp from TILT_PWR_MIN at 0° to 1.0 at TILT_PWR_FULL°.
# Replaces the old step function that hard-capped at 0.25 below 3° — starving the
# D term of authority and making small-angle corrections sluggish.
TILT_PWR_MIN   = 0.30   # power floor when perfectly upright
TILT_PWR_FULL  = 20.0   # degrees at which full power unlocks

# Sabertooth ignores PWM within ~±30 µs of 1500 (≈ ±0.06 in [-1,1] cmd space).
# Commands below this threshold produce zero torque.
MOTOR_DEADBAND = 0.06

CONTROL_HZ = 100
target_dt  = 1.0 / CONTROL_HZ

CMD_FORWARD    = 0.0
CMD_TURN       = 0.0
PITCH_TRIM_DEG = 0.0
PITCH_TRIM_RAD = math.radians(PITCH_TRIM_DEG)

# ---------------------------------------------------------------------------
# Wheel speed equalisation
# ---------------------------------------------------------------------------
# How many samples to average for each wheel (5 samples @ 100 Hz = 50 ms)
WHEEL_EQ_SMOOTH = 5

# Proportional gain on the speed difference — nudges the faster wheel down
# and the slower wheel up. Start at 0.02, increase if needed.
# Set to 0.0 to disable.
WHEEL_EQ_GAIN = 0.02

# Only apply correction when the average wheel speed is above this threshold
# (rad/s) — avoids fighting noise when the robot is nearly stationary.
WHEEL_EQ_MIN_SPEED = 0.3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def deadband(x, db):
    return 0.0 if abs(x) < db else x


def wrap_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _compensate_deadband(cmd):
    """Rescale so any non-trivial PID output jumps over the motor driver deadband.
    Maps [0.002, 1.0] → [MOTOR_DEADBAND, 1.0].  Below 0.002 → 0 (true stop)."""
    if abs(cmd) < 0.002:
        return 0.0
    return math.copysign(MOTOR_DEADBAND + abs(cmd) * (1.0 - MOTOR_DEADBAND), cmd)


def tilt_based_max_power(pitch_rad: float) -> float:
    pitch_deg = abs(math.degrees(pitch_rad))
    if pitch_deg >= TILT_PWR_FULL:
        return 1.0
    return TILT_PWR_MIN + (1.0 - TILT_PWR_MIN) * (pitch_deg / TILT_PWR_FULL)


# ---------------------------------------------------------------------------
# Background gains watcher — reads pid_gains.json every 1 s off the hot loop
# ---------------------------------------------------------------------------
_pending_gains: dict = {}
_pending_gains_lock  = _threading.Lock()


def _gains_watcher(path: str) -> None:
    while True:
        try:
            with open(path, "r") as _f:
                _g = _json.load(_f)
            with _pending_gains_lock:
                _pending_gains.update(_g)
        except Exception:
            pass
        time.sleep(1.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global KP_PITCH, KD_PITCH, KI_PITCH, PITCH_TRIM_RAD, PITCH_TIP_LIMIT

    # Convert SIGTERM (sent by the webapp's stop_program) into KeyboardInterrupt
    # so the finally block always runs — ensuring pigpio is closed cleanly and
    # motors are zeroed before the process exits.
    def _sigterm_handler(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _sigterm_handler)

    script_dir     = os.path.dirname(os.path.abspath(__file__))
    pid_gains_file = os.path.join(script_dir, "pid_gains.json")

    # Start background thread — reads pid_gains.json every 1 s, never blocks the loop
    _threading.Thread(target=_gains_watcher, args=(pid_gains_file,),
                      daemon=True, name="gains-watcher").start()
    log_path = os.path.join(
        script_dir,
        f"robot_pid_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )

    # PID state
    pitch_int          = 0.0
    drive_fwd          = 0.0
    drive_yaw          = 0.0
    heading_target     = None
    pitch_dot_filtered = 0.0

    # Wheel equalisation — rolling average buffers
    left_buf  = deque(maxlen=WHEEL_EQ_SMOOTH)
    right_buf = deque(maxlen=WHEEL_EQ_SMOOTH)

    loop_count = 0
    warn_count = 0
    last_time  = time.monotonic()

    log_file = open(log_path, "w", buffering=1)
    log_file.write(
        "timestamp,lin_vel,pitch_rad,pitch_rate,yaw_rate,"
        "wheel_v1,wheel_v2,left_cmd,right_cmd,eq_correction,dt_ms\n"
    )

    print("PIDLinear.py — Balance PID with smooth power curve + deadband compensation")
    print(f"   EQ gain: {WHEEL_EQ_GAIN}  min speed: {WHEEL_EQ_MIN_SPEED} rad/s  smooth: {WHEEL_EQ_SMOOTH} samples")
    print(f"   Tilt power: {TILT_PWR_MIN:.2f} @ 0°  →  1.0 @ {TILT_PWR_FULL:.0f}°")
    print(f"   Motor deadband compensation: {MOTOR_DEADBAND:.3f}")
    print(f"   Log -> {log_path}")
    print(f"   Loop -> {CONTROL_HZ} Hz  |  Ctrl+C to stop\n")

    try:
        while True:
            t_start   = time.monotonic()
            now       = t_start
            dt        = min(now - last_time, 0.2)
            last_time = now

            # 1) Read sensors directly from shared memory — no file I/O, no SQLite.
            obs_raw, age = _read_shm()
            if obs_raw is None or age > IMU_STALE_LIMIT_S:
                warn_count += 1
                if warn_count % 20 == 1:
                    print(f"\n[STALE] IMU SHM {age*1000:.0f} ms old — motors off (x{warn_count})")
                set_motor_velocities(0.0, 0.0)
                pitch_int      = 0.0
                heading_target = None
                left_buf.clear()
                right_buf.clear()
                elapsed = time.monotonic() - t_start
                if target_dt - elapsed > 0:
                    time.sleep(target_dt - elapsed)
                continue

            obs = np.asarray(obs_raw, dtype=np.float32)
            pitch           = float(obs[1] - PITCH_TRIM_RAD)
            pitch_dot_raw   = float(obs[2])
            pitch_dot_filtered = (GYRO_LPF_ALPHA * pitch_dot_raw
                                  + (1.0 - GYRO_LPF_ALPHA) * pitch_dot_filtered)
            pitch_dot       = pitch_dot_filtered
            yaw_rate        = float(obs[3])
            wheel_left_vel  = float(obs[4])
            wheel_right_vel = float(obs[5])
            yaw             = float(obs[8])

            # 2) Ramp drive_fwd
            target_fwd = CMD_FORWARD * FWD_MAX
            target_yaw = CMD_TURN    * YAW_MAX

            if target_fwd != 0.0:
                drive_fwd += math.copysign(FWD_ACCEL * dt, target_fwd - drive_fwd)
                drive_fwd  = clamp(drive_fwd,
                                   min(-target_fwd, target_fwd),
                                   max(-target_fwd, target_fwd))
            else:
                drive_fwd *= math.exp(-dt / FWD_DECAY)

            # 3) Ramp drive_yaw
            if target_yaw != 0.0:
                if heading_target is None:
                    heading_target = yaw
                drive_yaw = clamp(
                    drive_yaw + math.copysign(YAW_ACCEL * dt, target_yaw - drive_yaw),
                    -YAW_MAX, YAW_MAX,
                )
            else:
                drive_yaw = 0.0

            drive_fwd = deadband(clamp(drive_fwd, -FWD_MAX, FWD_MAX), 0.03)
            drive_yaw = clamp(drive_yaw, -YAW_MAX, YAW_MAX)

            # 4) Tip-over protection
            wheel_avg = clamp(
                0.5 * (wheel_left_vel + wheel_right_vel), -WHEEL_CMD_MAX, WHEEL_CMD_MAX
            )
            if abs(pitch) > PITCH_TIP_LIMIT:
                drive_fwd = 0.0; drive_yaw = 0.0; pitch_int = 0.0
                heading_target = None
                left_buf.clear(); right_buf.clear()
                set_motor_velocities(0.0, 0.0)
                loop_count += 1
                sys.stdout.write(
                    f"\r[{loop_count:6d}]  TIP-OVER  pitch={math.degrees(pitch):+6.2f}deg"
                    f"  motors=OFF   "
                )
                sys.stdout.flush()
                elapsed = time.monotonic() - t_start
                if target_dt - elapsed > 0:
                    time.sleep(target_dt - elapsed)
                continue

            # 5) Pitch balance PID
            pitch_error = pitch
            pitch_int   = clamp(pitch_int + pitch_error * dt, -I_PITCH_MAX, I_PITCH_MAX)
            wheel_balance_cmd = clamp(
                KP_PITCH * pitch_error
                + KD_PITCH * pitch_dot
                + KI_PITCH * pitch_int
                + K_WHEEL_AVG * wheel_avg,
                -WHEEL_CMD_MAX, WHEEL_CMD_MAX,
            )
            wheel_avg_cmd = wheel_balance_cmd + drive_fwd

            # 6) Yaw / heading control
            if target_yaw != 0.0:
                yaw_cmd        = drive_yaw
                heading_target = yaw
            else:
                if heading_target is None:
                    heading_target = yaw
                yaw_error = wrap_angle(yaw - heading_target)
                yaw_cmd   = clamp(
                    -(K_YAW_HOLD * yaw_error + K_YAW_RATE * yaw_rate),
                    -YAW_CMD_MAX, YAW_CMD_MAX,
                )

            # 6b) Cap yaw relative to balance so one motor never runs ≈2× the other
            _yaw_limit = max(abs(wheel_avg_cmd) * MAX_YAW_FRAC, MIN_YAW_ABS)
            yaw_cmd    = clamp(yaw_cmd, -_yaw_limit, _yaw_limit)

            # 7) Mix balance + yaw -> left / right
            left_cmd  = clamp(wheel_avg_cmd - yaw_cmd, -WHEEL_CMD_MAX, WHEEL_CMD_MAX) / WHEEL_CMD_MAX
            right_cmd = clamp(wheel_avg_cmd + yaw_cmd, -WHEEL_CMD_MAX, WHEEL_CMD_MAX) / WHEEL_CMD_MAX

            max_pwr   = tilt_based_max_power(pitch)
            left_cmd  = clamp(left_cmd,  -max_pwr, max_pwr)
            right_cmd = clamp(right_cmd, -max_pwr, max_pwr)

            # 8) Wheel speed equalisation
            eq_correction = 0.0
            if WHEEL_EQ_GAIN > 0.0:
                left_buf.append(wheel_left_vel)
                right_buf.append(wheel_right_vel)
                if len(left_buf) == WHEEL_EQ_SMOOTH and len(right_buf) == WHEEL_EQ_SMOOTH:
                    avg_left  = sum(left_buf)  / WHEEL_EQ_SMOOTH
                    avg_right = sum(right_buf) / WHEEL_EQ_SMOOTH
                    avg_speed = abs(avg_left + avg_right) / 2.0
                    if avg_speed > WHEEL_EQ_MIN_SPEED:
                        # Positive correction means left is faster — slow left, speed up right
                        eq_correction = clamp(
                            WHEEL_EQ_GAIN * (avg_left - avg_right),
                            -0.1, 0.1   # cap correction at ±10% to avoid instability
                        )
                        left_cmd  = clamp(left_cmd  - eq_correction, -max_pwr, max_pwr)
                        right_cmd = clamp(right_cmd + eq_correction, -max_pwr, max_pwr)

            left_cmd  = _compensate_deadband(left_cmd)
            right_cmd = _compensate_deadband(right_cmd)
            set_motor_velocities(left_cmd, right_cmd)

            # 9) Log
            t_now = time.monotonic()
            dt_ms = (t_now - t_start) * 1000.0
            log_file.write(
                f"{t_now:.6f},"
                f"{obs[0]:.6f},{pitch:.6f},{pitch_dot:.6f},{obs[3]:.6f},"
                f"{obs[4]:.6f},{obs[5]:.6f},"
                f"{left_cmd:.6f},{right_cmd:.6f},{eq_correction:.6f},{dt_ms:.3f}\n"
            )

            # 10) Apply any gains delivered by the background watcher (zero file I/O here)
            loop_count += 1
            _g = None
            with _pending_gains_lock:
                if _pending_gains:
                    _g = dict(_pending_gains)
                    _pending_gains.clear()
            if _g:
                KP_PITCH        = float(_g.get("kp",      KP_PITCH))
                KD_PITCH        = float(_g.get("kd",      KD_PITCH))
                KI_PITCH        = float(_g.get("ki",      KI_PITCH))
                PITCH_TRIM_RAD  = math.radians(float(_g.get("trim_deg", math.degrees(PITCH_TRIM_RAD))))
                PITCH_TIP_LIMIT = math.radians(float(_g.get("tip_deg",  math.degrees(PITCH_TIP_LIMIT))))

            # 11) Console
            sys.stdout.write(
                f"\r[{loop_count:6d}]  pitch={math.degrees(pitch):+6.2f}deg  "
                f"yaw={math.degrees(yaw):+7.2f}deg  "
                f"L={left_cmd:+.4f}  R={right_cmd:+.4f}  "
                f"eq={eq_correction:+.4f}  dt={dt_ms:.1f}ms   "
            )
            sys.stdout.flush()

            # 12) Sleep
            elapsed = time.monotonic() - t_start
            if target_dt - elapsed > 0:
                time.sleep(target_dt - elapsed)

    except KeyboardInterrupt:
        print(f"\nStopped after {loop_count} steps  ({warn_count} sensor warnings)")
    finally:
        set_motor_velocities(0.0, 0.0)
        close_motor_connection()
        log_file.close()
        if _shm_handle is not None:
            try:
                _shm_handle.close()
            except Exception:
                pass
        print(f"Motors stopped.  Log saved -> {log_path}")


if __name__ == "__main__":
    main()
