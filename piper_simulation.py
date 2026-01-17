#!/usr/bin/env python3
"""
Piper X Robotic Arm Simulation using MuJoCo

This script provides an interactive simulation of the Piper X robotic arm
that runs natively on macOS.

Usage (two options):

  Option 1 - Static viewer (regular python):
    source venv/bin/activate
    python piper_simulation.py

  Option 2 - Animated demo (requires mjpython):
    source venv/bin/activate
    mjpython piper_simulation.py --animate
"""

import sys
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path


# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MJCF_PATH = SCRIPT_DIR / "piper.xml"


def print_model_info(model: mujoco.MjModel):
    """Print information about the loaded model."""
    print("\n" + "="*60)
    print("PIPER X ROBOTIC ARM - Model Information")
    print("="*60)
    print(f"Number of bodies: {model.nbody}")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of DOF: {model.nv}")
    print(f"Number of actuators: {model.nu}")

    print("\nJoints:")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_type = model.jnt_type[i]
        type_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
        print(f"  [{i}] {joint_name}: {type_names.get(joint_type, 'unknown')}")

        if model.jnt_limited[i]:
            lower = model.jnt_range[i, 0]
            upper = model.jnt_range[i, 1]
            if joint_type == 3:  # hinge
                print(f"       Limits: [{np.degrees(lower):.1f}, {np.degrees(upper):.1f}] deg")
            else:  # slide
                print(f"       Limits: [{lower*1000:.1f}, {upper*1000:.1f}] mm")

    print("\nActuators:")
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  [{i}] {act_name}")

    print("="*60 + "\n")


class PiperController:
    """Controller for the Piper X robotic arm."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        self.arm_joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.arm_joint_ids = []
        for name in self.arm_joints:
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id >= 0:
                self.arm_joint_ids.append(jnt_id)

    def demo_motion(self, time: float) -> np.ndarray:
        """Generate smooth demo motion pattern."""
        freq = 0.2
        t = time
        return np.array([
            0.8 * np.sin(2 * np.pi * freq * t),
            1.2 + 0.4 * np.sin(2 * np.pi * freq * t + 0.5),
            -1.5 + 0.4 * np.sin(2 * np.pi * freq * t + 1.0),
            0.6 * np.sin(2 * np.pi * freq * t * 1.5),
            0.5 * np.sin(2 * np.pi * freq * t * 2.0),
            0.8 * np.sin(2 * np.pi * freq * t * 2.5),
        ])

    def control(self, model, data):
        """Control callback - called each simulation step."""
        target = self.demo_motion(data.time)

        for i, jnt_id in enumerate(self.arm_joint_ids):
            qpos_adr = model.jnt_qposadr[jnt_id]
            qvel_adr = model.jnt_dofadr[jnt_id]

            pos_error = target[i] - data.qpos[qpos_adr]
            vel_error = -data.qvel[qvel_adr]

            kp, kd = 100.0, 20.0
            if i < model.nu:
                data.ctrl[i] = kp * pos_error + kd * vel_error

        # Gripper motion
        gripper_target = 0.015 * (1 + np.sin(2 * np.pi * 0.3 * data.time))
        if model.nu > 6:
            data.ctrl[6] = gripper_target
        if model.nu > 7:
            data.ctrl[7] = -gripper_target


def set_initial_pose(model, data):
    """Set the arm to a reasonable starting pose."""
    initial_config = {
        "joint1": 0.0,
        "joint2": 1.0,
        "joint3": -1.5,
        "joint4": 0.0,
        "joint5": 0.0,
        "joint6": 0.0,
    }
    for joint_name, angle in initial_config.items():
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jnt_id >= 0:
            qpos_adr = model.jnt_qposadr[jnt_id]
            data.qpos[qpos_adr] = angle
    mujoco.mj_forward(model, data)


def run_animated(model, data):
    """Run with animated demo motion (requires mjpython on macOS)."""
    controller = PiperController(model, data)

    print("\nStarting ANIMATED simulation...")
    print("The arm will perform smooth demo motions.")
    print("-"*60)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            controller.control(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()


def run_static(model, data):
    """Run static viewer (works with regular python)."""
    print("\nStarting STATIC simulation...")
    print("Use mouse to interact. Physics runs but no demo motion.")
    print("For animated demo, run: mjpython piper_simulation.py --animate")
    print("-"*60)

    mujoco.viewer.launch(model, data)


def main():
    print("="*60)
    print("PIPER X ROBOTIC ARM SIMULATION")
    print("="*60)

    if not MJCF_PATH.exists():
        print(f"Error: Model file not found at {MJCF_PATH}")
        return

    print(f"\nLoading model from: {MJCF_PATH}")
    try:
        model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    data = mujoco.MjData(model)
    print_model_info(model)

    print("Setting initial pose...")
    mujoco.mj_resetData(model, data)
    set_initial_pose(model, data)

    print("\nControls:")
    print("  Mouse drag     - Rotate view")
    print("  Shift + drag   - Pan view")
    print("  Scroll         - Zoom")
    print("  Double-click   - Select body")
    print("  Ctrl + drag    - Apply force to selected body")
    print("  Space          - Pause/Resume")
    print("  Backspace      - Reset simulation")
    print("  ESC            - Exit")

    # Check for --animate flag
    if "--animate" in sys.argv:
        run_animated(model, data)
    else:
        run_static(model, data)


if __name__ == "__main__":
    main()
