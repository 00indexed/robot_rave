#!/usr/bin/env python3
"""
Piper X Dual Arm Control - Two robots waving together!

Usage:
    mjpython piper_dual_control.py
"""

import sys
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path


MJCF_PATH = Path(__file__).parent / "piper_dual.xml"


class DualPiperArms:
    """Control two Piper X arms simultaneously."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
        self.data = mujoco.MjData(self.model)

        # Joint names for each arm
        self.left_joints = [f"L_joint{i}" for i in range(1, 7)]
        self.right_joints = [f"R_joint{i}" for i in range(1, 7)]

        # Get joint IDs
        self.left_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self.left_joints
        ]
        self.right_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self.right_joints
        ]

        # Target positions for each arm (6 joints each)
        self.left_target = np.zeros(6)
        self.right_target = np.zeros(6)

        # Gripper targets
        self.left_gripper = 0.0
        self.right_gripper = 0.0

        # Initialize
        mujoco.mj_resetData(self.model, self.data)

    def set_left_arm(self, positions: list):
        """Set target positions for left arm."""
        self.left_target = np.array(positions)

    def set_right_arm(self, positions: list):
        """Set target positions for right arm."""
        self.right_target = np.array(positions)

    def open_left_gripper(self):
        self.left_gripper = 0.035

    def close_left_gripper(self):
        self.left_gripper = 0.0

    def open_right_gripper(self):
        self.right_gripper = 0.035

    def close_right_gripper(self):
        self.right_gripper = 0.0

    def control_step(self):
        """Apply control to both arms."""
        # Left arm: actuators 0-7
        for i in range(6):
            self.data.ctrl[i] = self.left_target[i]
        self.data.ctrl[6] = self.left_gripper
        self.data.ctrl[7] = -self.left_gripper

        # Right arm: actuators 8-15
        for i in range(6):
            self.data.ctrl[8 + i] = self.right_target[i]
        self.data.ctrl[14] = self.right_gripper
        self.data.ctrl[15] = -self.right_gripper

    def step(self):
        """Step simulation with control."""
        self.control_step()
        mujoco.mj_step(self.model, self.data)

    @property
    def time(self):
        return self.data.time


def dual_wave_hello(arms: DualPiperArms, t: float):
    """Both arms wave hello together!"""
    wave_freq = 0.5

    # Left arm wave (slightly offset timing)
    left_wave = np.sin(2 * np.pi * wave_freq * t)
    arms.set_left_arm([
        0.0,                    # base
        1.8,                    # shoulder up
        -2.5,                   # elbow bent
        0.0,                    # wrist1
        0.5 * left_wave,        # wrist2 - wave!
        0.0,                    # wrist3
    ])
    arms.open_left_gripper()

    # Right arm wave (opposite phase for variety)
    right_wave = np.sin(2 * np.pi * wave_freq * t + np.pi)  # opposite phase
    arms.set_right_arm([
        0.0,                    # base
        1.8,                    # shoulder up
        -2.5,                   # elbow bent
        0.0,                    # wrist1
        0.5 * right_wave,       # wrist2 - wave!
        0.0,                    # wrist3
    ])
    arms.open_right_gripper()


def synchronized_wave(arms: DualPiperArms, t: float):
    """Both arms wave in perfect sync."""
    wave_freq = 0.5
    wave = np.sin(2 * np.pi * wave_freq * t)

    pose = [0.0, 1.8, -2.5, 0.0, 0.5 * wave, 0.0]

    arms.set_left_arm(pose)
    arms.set_right_arm(pose)
    arms.open_left_gripper()
    arms.open_right_gripper()


def mirror_dance(arms: DualPiperArms, t: float):
    """Arms mirror each other in a dance."""
    freq = 0.3

    # Base oscillation
    base = 0.4 * np.sin(2 * np.pi * freq * t)
    shoulder = 1.5 + 0.3 * np.sin(2 * np.pi * freq * t)
    elbow = -2.0 + 0.3 * np.cos(2 * np.pi * freq * t)
    wrist = 0.4 * np.sin(2 * np.pi * freq * 2 * t)

    # Left arm
    arms.set_left_arm([base, shoulder, elbow, 0, wrist, 0])

    # Right arm mirrors (base rotation inverted)
    arms.set_right_arm([-base, shoulder, elbow, 0, -wrist, 0])

    arms.open_left_gripper()
    arms.open_right_gripper()


def robot_clap(arms: DualPiperArms, t: float):
    """Arms come together to clap!"""
    freq = 0.4
    cycle = t * freq
    phase = cycle % 1.0

    if phase < 0.5:
        # Arms apart
        left_base = 0.5
        right_base = -0.5
        gripper = 0.035
    else:
        # Arms together (clap!)
        left_base = 0.1
        right_base = -0.1
        gripper = 0.0

    arms.set_left_arm([left_base, 1.2, -1.8, 0, 0, 0])
    arms.set_right_arm([right_base, 1.2, -1.8, 0, 0, 0])

    arms.left_gripper = gripper
    arms.right_gripper = gripper


def main():
    print("="*60)
    print("PIPER X DUAL ARM SIMULATION")
    print("="*60)

    if not MJCF_PATH.exists():
        print(f"Error: {MJCF_PATH} not found")
        return

    print(f"\nLoading dual arm model...")
    arms = DualPiperArms()

    print(f"Model loaded: {arms.model.njnt} joints, {arms.model.nu} actuators")

    # Select motion pattern
    motion_funcs = {
        "wave": dual_wave_hello,
        "sync": synchronized_wave,
        "mirror": mirror_dance,
        "clap": robot_clap,
    }

    mode = "wave"
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            mode = sys.argv[idx + 1]

    if mode not in motion_funcs:
        print(f"Unknown mode '{mode}'. Available: {list(motion_funcs.keys())}")
        return

    motion_func = motion_funcs[mode]

    print(f"\nRunning '{mode}' demo with two arms!")
    print("Close the viewer to exit.")
    print("\nControls:")
    print("  Mouse drag - Rotate view")
    print("  Scroll     - Zoom")
    print("  Space      - Pause")
    print("-"*60)

    with mujoco.viewer.launch_passive(arms.model, arms.data) as viewer:
        while viewer.is_running():
            motion_func(arms, arms.time)
            arms.step()
            viewer.sync()


if __name__ == "__main__":
    main()
