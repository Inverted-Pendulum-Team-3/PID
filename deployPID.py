#!/usr/bin/env python3
"""
deployPID_NEW_hw.py — Dylan's PID algorithm wired to Arian's hardware interface.

Control logic from DeployPID_NEW.py:
  - Pitch PD balance loop
  - Heading hold when not turning (yaw integration for relative heading)
  - Forward drive with acceleration ramp and exponential decay
  - Hard tip-over cutoff at ±70°

Hardware wiring:
  - get_sensor_data()        → reads obs_cache.bin or SQLite (from sensorsHI.py)
  - set_motor_velocities()   → sends PWM to Sabertooth via pigpio
  - close_motor_connection() → safe stop on exit

Observation vector (obs):
  [0] linear_velocity (m/s)
  [1] pitch (rad)
  [2] pitch_rate (rad/s)
  [3] yaw_rate (rad/s)
  [4] wheel_velocity_1, [5] wheel_velocity_2
  [6] velocity_error, [7] rotation_error, [8] yaw (rad, absolute from IMU)

Run: python3 deployPID_NEW_hw.py
"""

import os
import sys
import math
import time
from datetime import datetime

import numpy as np

from hardware_interface import (
    get_sensor_data,
    set_motor_velocities,
    close_motor_connection,
)

# ---------------------------------------------------------------------------
# PID / control gains (from DeployPID_NEW.py — tune these on hardware)
# ---------------------------------------------------------------------------
KP_PITCH      = 50.0
KD_PITCH      =  8.0
KI_PITCH      =  0.0
I_PITCH_MAX   =  2.0
WHEEL_CMD_MAX = 12.8   # internal command range before normalising to ±1
K_WHEEL_AVG   =  0.0   # wheel-speed feed-forward weight (set > 0 to use)

FWD_MAX       =  2.0
FWD_ACCEL     = 10.0
FWD_DECAY     =  0.5   # exponential decay time constant when no forward cmd

YAW_MAX       =  2.0
YAW_ACCEL     = 10.0

K_YAW_HOLD    =  1.0   # heading-hold proportional gain
K_YAW_RATE    =  0.2   # yaw-rate damping gain
YAW_CMD_MAX   =  1.0

PITCH_TIP_LIMIT = math.radians(70)  # cut motors beyond ±70°

CONTROL_HZ = 100
target_dt  = 1.0 / CONTROL_HZ

# User-commanded inputs (0 = stationary/straight)
# Change these or wire to a joystick / web app later.
CMD_FORWARD = 0.0
CMD_TURN    = 0.0


# ---------------------------------------------------------------------------
# Helpers (unchanged from DeployPID_NEW.py)
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


def tilt_based_max_power(pitch_rad: float) -> float:
    """Gradually limit motor power near level to prevent overshoot on small corrections."""
    pitch_deg = abs(math.degrees(pitch_rad))
    if pitch_deg <= 3.0:
        return 0.25
    if pitch_deg <= 8.0:
        return 0.55
    if pitch_deg <= 15.0:
        return 0.85
    return 1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global LOOP_TEST_VAL, KP_PITCH

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(
        script_dir,
        f"robot_pid_new_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    )

    # PID state
    pitch_int      = 0.0
    drive_fwd      = 0.0
    drive_yaw      = 0.0
    heading_target = None    # absolute yaw heading to hold (rad, from IMU)

    loop_count = 0
    warn_count = 0
    last_time  = time.monotonic()

    log_file = open(log_path, "w", buffering=1)
    log_file.write(
        "timestamp,lin_vel,pitch_rad,pitch_rate,yaw_rate,"
        "wheel_v1,wheel_v2,left_cmd,right_cmd,dt_ms\n"
    )

    print("✅ PID_NEW (Dylan) controller loaded")
    print(f"   Log → {log_path}")
    print(f"🚀 Control loop at {CONTROL_HZ} Hz  (Ctrl+C to stop)\n")

    try:
        while True:
            t_start   = time.monotonic()
            now       = t_start
            dt        = min(now - last_time, 0.2)
            last_time = now

            # ------------------------------------------------------------------
            # 1. Read sensors
            # ------------------------------------------------------------------
            obs = get_sensor_data(
                target_velocity      = CMD_FORWARD * FWD_MAX,
                target_rotation_rate = CMD_TURN    * YAW_MAX,
            )
            if obs is None:
                warn_count += 1
                if warn_count % 50 == 1:
                    print(f"[Warning] No sensor data (×{warn_count}), waiting...")
                time.sleep(0.005)
                continue

            obs = np.asarray(obs, dtype=np.float32)

            pitch           = float(obs[1])   # rad
            pitch_dot       = float(obs[2])   # pitch_rate rad/s
            yaw_rate        = float(obs[3])   # rad/s (from IMU gyro Z)
            wheel_left_vel  = float(obs[4])
            wheel_right_vel = float(obs[5])
            yaw             = float(obs[8])   # absolute yaw from IMU quaternion (rad)

            # ------------------------------------------------------------------
            # 2. Ramp drive_fwd toward target
            # ------------------------------------------------------------------
            target_fwd = CMD_FORWARD * FWD_MAX
            target_yaw = CMD_TURN    * YAW_MAX

            if target_fwd != 0.0:
                drive_fwd += math.copysign(FWD_ACCEL * dt, target_fwd - drive_fwd)
                if target_fwd > 0:
                    drive_fwd = clamp(drive_fwd, -target_fwd, target_fwd)
                else:
                    drive_fwd = clamp(drive_fwd, target_fwd, -target_fwd)
            else:
                drive_fwd *= math.exp(-dt / FWD_DECAY)

            # ------------------------------------------------------------------
            # 3. Ramp drive_yaw toward target
            # ------------------------------------------------------------------
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

            # ------------------------------------------------------------------
            # 4. Tip-over protection
            # ------------------------------------------------------------------
            wheel_avg = clamp(
                0.5 * (wheel_left_vel + wheel_right_vel), -WHEEL_CMD_MAX, WHEEL_CMD_MAX
            )

            if abs(pitch) > PITCH_TIP_LIMIT:
                drive_fwd      = 0.0
                drive_yaw      = 0.0
                pitch_int      = 0.0
                heading_target = None
                set_motor_velocities(0.0, 0.0)
                loop_count += 1
                continue

            # ------------------------------------------------------------------
            # 5. Pitch balance PID
            # ------------------------------------------------------------------
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

            # ------------------------------------------------------------------
            # 6. Yaw / heading control
            # ------------------------------------------------------------------
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

            # ------------------------------------------------------------------
            # 7. Mix balance + yaw → left / right in [-1, 1]
            # ------------------------------------------------------------------
            left_cmd  = clamp(wheel_avg_cmd - yaw_cmd, -WHEEL_CMD_MAX, WHEEL_CMD_MAX) / WHEEL_CMD_MAX
            right_cmd = clamp(wheel_avg_cmd + yaw_cmd, -WHEEL_CMD_MAX, WHEEL_CMD_MAX) / WHEEL_CMD_MAX

            # Tilt-based power cap: reduce max output near level to prevent overshoot
            max_pwr   = tilt_based_max_power(pitch)
            left_cmd  = clamp(left_cmd,  -max_pwr, max_pwr)
            right_cmd = clamp(right_cmd, -max_pwr, max_pwr)

            set_motor_velocities(left_cmd, right_cmd)

            # ------------------------------------------------------------------
            # 8. Log
            # ------------------------------------------------------------------
            t_now = time.monotonic()
            dt_ms = (t_now - t_start) * 1000.0
            log_file.write(
                f"{t_now:.6f},"
                f"{obs[0]:.6f},{obs[1]:.6f},{obs[2]:.6f},{obs[3]:.6f},"
                f"{obs[4]:.6f},{obs[5]:.6f},"
                f"{left_cmd:.6f},{right_cmd:.6f},{dt_ms:.3f}\n"
            )

            # ------------------------------------------------------------------
            # 9. Live stationary console line (updates every loop)
            # ------------------------------------------------------------------
            loop_count += 1
            sys.stdout.write(
                f"\r[{loop_count:6d}]  pitch={math.degrees(pitch):+6.2f}°  "
                f"yaw={math.degrees(yaw):+7.2f}°  "
                f"yaw_rate={math.degrees(yaw_rate):+6.2f}°/s  "
                f"vel={float(obs[0]):+.3f} m/s  "
                f"L={left_cmd:+.4f}  R={right_cmd:+.4f}  "
                f"dt={dt_ms:.1f}ms   "
            )
            sys.stdout.flush()

            # ------------------------------------------------------------------
            # 10. Sleep to maintain loop rate
            # ------------------------------------------------------------------
            elapsed = time.monotonic() - t_start
            sleep_t = target_dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print(f"\n⛔ Stopped after {loop_count} steps  ({warn_count} sensor warnings)")
    finally:
        set_motor_velocities(0.0, 0.0)
        close_motor_connection()
        log_file.close()
        print(f"✅ Motors stopped.  Log saved → {log_path}")


if __name__ == "__main__":
    main()
