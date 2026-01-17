#!/usr/bin/env python3
"""
Piper X Control Interface

This module provides a clean API for controlling the Piper X arm in simulation.
You can import this in your own scripts or run it directly for demos.

Usage:
    mjpython piper_control.py              # Run interactive demo
    mjpython piper_control.py --mode wave  # Wave motion
    mjpython piper_control.py --mode pick  # Pick and place demo
"""

import sys
import time
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, List
from collections import deque


MJCF_PATH = Path(__file__).parent / "piper.xml"
LOG_FILE = Path(__file__).parent / "motion_log.txt"


@dataclass
class MotionFrame:
    """Single frame of motion telemetry."""
    time: float
    target_pos: np.ndarray
    actual_pos: np.ndarray
    velocity: np.ndarray
    torque: np.ndarray
    pos_error: np.ndarray

    def __str__(self):
        return (f"t={self.time:6.3f} | "
                f"err=[{', '.join(f'{e:+6.3f}' for e in self.pos_error)}] | "
                f"vel=[{', '.join(f'{v:+6.2f}' for v in self.velocity)}] | "
                f"torque=[{', '.join(f'{t:+7.1f}' for t in self.torque)}]")


class MotionLogger:
    """Logs motion telemetry for analysis."""

    def __init__(self, max_frames: int = 5000):
        self.frames: deque = deque(maxlen=max_frames)
        self.enabled = True

    def log(self, time: float, target: np.ndarray, actual: np.ndarray,
            velocity: np.ndarray, torque: np.ndarray):
        if not self.enabled:
            return
        frame = MotionFrame(
            time=time,
            target_pos=target.copy(),
            actual_pos=actual.copy(),
            velocity=velocity.copy(),
            torque=torque.copy(),
            pos_error=target - actual
        )
        self.frames.append(frame)

    def save(self, filepath: Path = LOG_FILE):
        """Save log to file for analysis."""
        with open(filepath, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("PIPER X MOTION LOG\n")
            f.write("=" * 100 + "\n\n")

            if not self.frames:
                f.write("No frames logged.\n")
                return

            # Summary statistics
            errors = np.array([fr.pos_error for fr in self.frames])
            velocities = np.array([fr.velocity for fr in self.frames])
            torques = np.array([fr.torque for fr in self.frames])

            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total frames: {len(self.frames)}\n")
            f.write(f"Time range: {self.frames[0].time:.3f}s to {self.frames[-1].time:.3f}s\n\n")

            f.write("Position Error (radians) - per joint:\n")
            f.write(f"  {'Joint':<8} {'Mean':>10} {'Std':>10} {'Max':>10} {'Min':>10}\n")
            for i in range(6):
                err = errors[:, i]
                f.write(f"  Joint {i+1}  {np.mean(err):>+10.4f} {np.std(err):>10.4f} "
                       f"{np.max(err):>+10.4f} {np.min(err):>+10.4f}\n")

            f.write("\nVelocity (rad/s) - per joint:\n")
            f.write(f"  {'Joint':<8} {'Mean':>10} {'Std':>10} {'Max':>10} {'Min':>10}\n")
            for i in range(6):
                vel = velocities[:, i]
                f.write(f"  Joint {i+1}  {np.mean(vel):>+10.4f} {np.std(vel):>10.4f} "
                       f"{np.max(vel):>+10.4f} {np.min(vel):>+10.4f}\n")

            f.write("\nTorque (Nm) - per joint:\n")
            f.write(f"  {'Joint':<8} {'Mean':>10} {'Std':>10} {'Max':>10} {'Min':>10}\n")
            for i in range(6):
                trq = torques[:, i]
                f.write(f"  Joint {i+1}  {np.mean(trq):>+10.1f} {np.std(trq):>10.1f} "
                       f"{np.max(trq):>+10.1f} {np.min(trq):>+10.1f}\n")

            # Detect jitter - high frequency velocity changes
            f.write("\n" + "=" * 50 + "\n")
            f.write("JITTER ANALYSIS\n")
            f.write("-" * 50 + "\n")

            vel_diff = np.diff(velocities, axis=0)
            accel_magnitude = np.linalg.norm(vel_diff, axis=1)

            jitter_threshold = 5.0  # rad/s^2 change between frames
            jitter_frames = np.sum(accel_magnitude > jitter_threshold)
            jitter_pct = 100 * jitter_frames / len(accel_magnitude) if len(accel_magnitude) > 0 else 0

            f.write(f"Velocity change threshold: {jitter_threshold} rad/s per step\n")
            f.write(f"Frames with high acceleration: {jitter_frames} ({jitter_pct:.1f}%)\n")
            f.write(f"Mean acceleration magnitude: {np.mean(accel_magnitude):.4f}\n")
            f.write(f"Max acceleration magnitude: {np.max(accel_magnitude):.4f}\n")

            # Check for oscillation (sign changes in error)
            f.write("\nOscillation detection (error sign changes):\n")
            for i in range(6):
                err = errors[:, i]
                sign_changes = np.sum(np.diff(np.sign(err)) != 0)
                f.write(f"  Joint {i+1}: {sign_changes} sign changes "
                       f"({100*sign_changes/len(err):.1f}% of frames)\n")

            # Worst frames
            f.write("\n" + "=" * 50 + "\n")
            f.write("WORST FRAMES (by total position error)\n")
            f.write("-" * 50 + "\n")

            total_errors = np.linalg.norm(errors, axis=1)
            worst_indices = np.argsort(total_errors)[-10:][::-1]

            for idx in worst_indices:
                f.write(f"{self.frames[idx]}\n")

            # Recent frames
            f.write("\n" + "=" * 50 + "\n")
            f.write("LAST 50 FRAMES\n")
            f.write("-" * 50 + "\n")

            for frame in list(self.frames)[-50:]:
                f.write(f"{frame}\n")

        print(f"\nMotion log saved to: {filepath}")
        return filepath

    def get_diagnostics(self) -> dict:
        """Get diagnostic summary."""
        if not self.frames:
            return {"status": "no data"}

        errors = np.array([fr.pos_error for fr in self.frames])
        velocities = np.array([fr.velocity for fr in self.frames])

        return {
            "frames": len(self.frames),
            "mean_error": np.mean(np.abs(errors)),
            "max_error": np.max(np.abs(errors)),
            "mean_velocity": np.mean(np.abs(velocities)),
            "max_velocity": np.max(np.abs(velocities)),
        }


@dataclass
class JointLimits:
    """Joint limits in radians."""
    joint1: tuple = (-2.618, 2.618)   # Base rotation: ±150°
    joint2: tuple = (0, 3.14)          # Shoulder: 0° to 180°
    joint3: tuple = (-2.967, 0)        # Elbow: -170° to 0°
    joint4: tuple = (-1.745, 1.745)    # Wrist1: ±100°
    joint5: tuple = (-1.22, 1.22)      # Wrist2: ±70°
    joint6: tuple = (-2.094, 2.094)    # Wrist3: ±120°


class PiperArm:
    """
    High-level interface for controlling the Piper X arm.

    Example:
        arm = PiperArm()
        arm.move_to_joint_positions([0, 1.0, -1.5, 0, 0, 0])
        arm.open_gripper()
        arm.close_gripper()
    """

    def __init__(self, enable_logging: bool = True):
        self.model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
        self.data = mujoco.MjData(self.model)

        # Joint names
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.gripper_names = ["joint7", "joint8"]

        # Get joint IDs
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                         for n in self.joint_names]

        # Control gains (TUNED for stability)
        # Lower Kp = less aggressive response
        # Higher Kd = more damping to prevent oscillation
        self.kp = np.array([40, 60, 50, 30, 25, 20])   # Position gains (reduced)
        self.kd = np.array([15, 25, 20, 12, 10, 8])    # Velocity/damping gains

        # Target positions
        self.target_joints = np.zeros(6)
        self.target_gripper = 0.0  # 0 = closed, 0.035 = open

        # Last applied torques (for logging)
        self._last_torques = np.zeros(6)

        # Limits
        self.limits = JointLimits()

        # Motion logger
        self.logger = MotionLogger() if enable_logging else None

        # Initialize
        mujoco.mj_resetData(self.model, self.data)
        self.home()

    def home(self):
        """Move to home position."""
        self.target_joints = np.array([0.0, 1.0, -1.5, 0.0, 0.0, 0.0])
        self._apply_positions(self.target_joints)

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions in radians."""
        positions = []
        for jid in self.joint_ids:
            qpos_adr = self.model.jnt_qposadr[jid]
            positions.append(self.data.qpos[qpos_adr])
        return np.array(positions)

    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities."""
        velocities = []
        for jid in self.joint_ids:
            qvel_adr = self.model.jnt_dofadr[jid]
            velocities.append(self.data.qvel[qvel_adr])
        return np.array(velocities)

    def get_end_effector_position(self) -> np.ndarray:
        """Get end effector (gripper) position in world coordinates."""
        gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        return self.data.xpos[gripper_id].copy()

    def get_end_effector_orientation(self) -> np.ndarray:
        """Get end effector orientation as rotation matrix."""
        gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        return self.data.xmat[gripper_id].reshape(3, 3).copy()

    def move_to_joint_positions(self, positions: list, duration: float = 1.0):
        """
        Set target joint positions (in radians).

        Args:
            positions: List of 6 joint angles [j1, j2, j3, j4, j5, j6]
            duration: Not used in direct control, but hints at motion speed
        """
        self.target_joints = np.array(positions)
        # Clamp to limits
        limits = [self.limits.joint1, self.limits.joint2, self.limits.joint3,
                  self.limits.joint4, self.limits.joint5, self.limits.joint6]
        for i, (lo, hi) in enumerate(limits):
            self.target_joints[i] = np.clip(self.target_joints[i], lo, hi)

    def open_gripper(self):
        """Open the gripper fully."""
        self.target_gripper = 0.035

    def close_gripper(self):
        """Close the gripper fully."""
        self.target_gripper = 0.0

    def set_gripper(self, opening: float):
        """Set gripper opening (0.0 = closed, 0.035 = fully open)."""
        self.target_gripper = np.clip(opening, 0.0, 0.035)

    def _apply_positions(self, positions: np.ndarray):
        """Directly set joint positions (for initialization)."""
        for i, jid in enumerate(self.joint_ids):
            qpos_adr = self.model.jnt_qposadr[jid]
            self.data.qpos[qpos_adr] = positions[i]
        mujoco.mj_forward(self.model, self.data)

    def control_step(self):
        """
        Set target positions for position servos.
        MuJoCo's position actuators handle the PD control internally.
        """
        current_pos = self.get_joint_positions()
        current_vel = self.get_joint_velocities()

        # Position servos: just set the target position directly
        # MuJoCo handles the PD control internally
        for i in range(min(6, self.model.nu)):
            self.data.ctrl[i] = self.target_joints[i]

        # Gripper control
        if self.model.nu > 6:
            self.data.ctrl[6] = self.target_gripper
        if self.model.nu > 7:
            self.data.ctrl[7] = -self.target_gripper

        # Log telemetry
        if self.logger:
            self.logger.log(
                time=self.data.time,
                target=self.target_joints,
                actual=current_pos,
                velocity=current_vel,
                torque=self.target_joints  # Now this is target pos, not torque
            )

    def save_log(self, filepath: Optional[Path] = None):
        """Save motion log to file."""
        if self.logger:
            return self.logger.save(filepath or LOG_FILE)
        return None

    def step(self):
        """Advance simulation one step with control."""
        self.control_step()
        mujoco.mj_step(self.model, self.data)

    @property
    def time(self) -> float:
        """Current simulation time."""
        return self.data.time


# ============== Demo Motion Patterns ==============

def wave_motion(arm: PiperArm, t: float):
    """Smooth waving motion - slower frequency for stable tracking."""
    freq = 0.15  # Hz - slow enough for arm to follow
    arm.move_to_joint_positions([
        0.6 * np.sin(2 * np.pi * freq * t),                # base swings side to side
        1.0 + 0.25 * np.sin(2 * np.pi * freq * t),         # shoulder bobs
        -1.5 + 0.25 * np.cos(2 * np.pi * freq * t),        # elbow waves
        0.4 * np.sin(2 * np.pi * freq * 1.2 * t),          # wrist twist
        0.3 * np.sin(2 * np.pi * freq * 1.5 * t),          # wrist nod
        0.5 * np.sin(2 * np.pi * freq * 1.8 * t),          # wrist roll
    ])


def pick_and_place_motion(arm: PiperArm, t: float):
    """Simulated pick and place cycle - slower transitions."""
    cycle_time = 12.0  # seconds per cycle (slower)
    phase = (t % cycle_time) / cycle_time

    if phase < 0.15:
        # Move to pick position
        arm.move_to_joint_positions([0.4, 1.4, -1.8, 0, 0.4, 0])
        arm.open_gripper()
    elif phase < 0.3:
        # Lower and close gripper
        arm.move_to_joint_positions([0.4, 1.6, -2.0, 0, 0.4, 0])
        arm.close_gripper()
    elif phase < 0.45:
        # Lift
        arm.move_to_joint_positions([0.4, 1.2, -1.5, 0, 0.3, 0])
    elif phase < 0.65:
        # Move to place position
        arm.move_to_joint_positions([-0.4, 1.2, -1.5, 0, 0.3, 0])
    elif phase < 0.8:
        # Lower and release
        arm.move_to_joint_positions([-0.4, 1.6, -2.0, 0, 0.4, 0])
        arm.open_gripper()
    else:
        # Return to ready
        arm.move_to_joint_positions([0, 1.0, -1.5, 0, 0, 0])


def circle_motion(arm: PiperArm, t: float):
    """End effector traces a circle (approximately) - slow and smooth."""
    freq = 0.1  # Hz - slow circular motion

    arm.move_to_joint_positions([
        0.5 * np.sin(2 * np.pi * freq * t),        # base follows circle
        1.0 + 0.2 * np.sin(2 * np.pi * freq * t),  # shoulder adjusts height
        -1.5 + 0.2 * np.cos(2 * np.pi * freq * t), # elbow reaches
        0,
        0.25 * np.cos(2 * np.pi * freq * t),       # wrist compensates
        0,
    ])


def wave_hello(arm: PiperArm, t: float):
    """Wave hello like a friendly robot - slow, natural waving motion."""
    wave_freq = 0.5  # Hz - friendly wave speed

    # Smooth wave motion using sine
    wave = np.sin(2 * np.pi * wave_freq * t)

    # Arm raised up like waving hello
    # Joint angles tuned for "hand up, waving" pose
    arm.move_to_joint_positions([
        0.0,                          # base: facing forward
        1.8,                          # shoulder: arm angled up
        -2.5,                         # elbow: bent back so gripper is up high
        0.0,                          # wrist1: neutral
        0.5 * wave,                   # wrist2: THIS is the wave motion (side to side)
        0.0,                          # wrist3: neutral
    ])

    # Keep gripper open (like an open hand waving)
    arm.open_gripper()


def keyboard_control_demo(arm: PiperArm):
    """
    Print instructions for keyboard-style control.
    (Note: MuJoCo viewer doesn't expose keyboard in passive mode easily,
    but you can modify target positions programmatically)
    """
    print("\n" + "="*60)
    print("PROGRAMMATIC CONTROL DEMO")
    print("="*60)
    print("""
The PiperArm class provides these methods:

  arm.move_to_joint_positions([j1, j2, j3, j4, j5, j6])
      Set target angles in radians for all 6 joints

  arm.open_gripper() / arm.close_gripper()
      Control the gripper

  arm.get_joint_positions()
      Returns current joint angles as numpy array

  arm.get_end_effector_position()
      Returns [x, y, z] position of gripper in world frame

Joint Ranges (in degrees):
  Joint 1 (base):     ±150°
  Joint 2 (shoulder): 0° to 180°
  Joint 3 (elbow):    -170° to 0°
  Joint 4 (wrist1):   ±100°
  Joint 5 (wrist2):   ±70°
  Joint 6 (wrist3):   ±120°

Example - Move to a specific pose:
  arm.move_to_joint_positions([0, 1.57, -1.57, 0, 0, 0])

Example - Read position:
  pos = arm.get_end_effector_position()
  print(f"Gripper at: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
""")
    print("="*60)


def run_demo(mode: str = "wave"):
    """Run a demo with the specified motion pattern."""
    arm = PiperArm(enable_logging=True)

    motion_funcs = {
        "wave": wave_motion,
        "pick": pick_and_place_motion,
        "circle": circle_motion,
        "hello": wave_hello,
    }

    motion_func = motion_funcs.get(mode, wave_motion)

    print(f"\nRunning '{mode}' demo...")
    print("Close the viewer window to exit and save motion log.")
    keyboard_control_demo(arm)

    try:
        with mujoco.viewer.launch_passive(arm.model, arm.data) as viewer:
            while viewer.is_running():
                # Update motion target
                motion_func(arm, arm.time)

                # Step simulation with control
                arm.step()

                # Sync viewer
                viewer.sync()
    finally:
        # Always save log when exiting
        print("\nSaving motion log...")
        arm.save_log()

        # Print quick diagnostics
        if arm.logger:
            diag = arm.logger.get_diagnostics()
            print(f"\nQuick diagnostics:")
            print(f"  Frames logged: {diag.get('frames', 0)}")
            print(f"  Mean position error: {diag.get('mean_error', 0):.4f} rad")
            print(f"  Max position error: {diag.get('max_error', 0):.4f} rad")
            print(f"  Max velocity: {diag.get('max_velocity', 0):.2f} rad/s")


def main():
    # Parse command line
    mode = "wave"
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            mode = sys.argv[idx + 1]

    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print("\nAvailable modes: wave, pick, circle, hello")
        return

    run_demo(mode)


if __name__ == "__main__":
    main()
