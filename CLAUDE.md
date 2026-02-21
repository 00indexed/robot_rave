# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Clone Piper ROS package for URDF/mesh assets (required)
git clone https://github.com/agilexrobotics/piper_ros.git
```

### Running Simulations
All simulations use `mjpython` (MuJoCo's Python runtime):

```bash
# Single arm demos
mjpython piper_control.py --mode hello   # Wave hello
mjpython piper_control.py --mode wave    # Smooth wave motion
mjpython piper_control.py --mode pick    # Pick and place
mjpython piper_control.py --mode circle  # Circular motion

# Dual arm demos
mjpython piper_dual_control.py --mode wave    # Both arms wave
mjpython piper_dual_control.py --mode sync    # Synchronized waving
mjpython piper_dual_control.py --mode mirror  # Mirror dance
mjpython piper_dual_control.py --mode clap    # Arms clap together

# Vision-based tracking (webcam)
mjpython piper_vision_control.py                              # Track right arm
mjpython piper_vision_control.py --side left --no-mirror      # Track left arm
mjpython piper_vision_control.py --smoothing 0.7 --max-step 0.35  # Tune responsiveness

# Basic simulation (minimal)
mjpython piper_simulation.py
```

### Viewer Controls
- Mouse drag: Rotate view
- Shift + drag: Pan view
- Scroll: Zoom
- Space: Pause/Resume
- Backspace: Reset
- ESC: Exit

## Architecture

### Core Components

**PiperArm** (`piper_control.py`)
- Main control interface for single arm simulation
- Uses position servo control with tuned PD gains (kp/kd arrays)
- Provides high-level API: `move_to_joint_positions()`, `open_gripper()`, `close_gripper()`
- Encapsulates joint limits, motion logging, and telemetry
- 6 DOF arm joints + 2 DOF gripper

**DualPiperArms** (`piper_dual_control.py`)
- Controls two independent Piper X arms
- Each arm has separate target positions and gripper control
- Actuator indices: Left arm (0-7), Right arm (8-15)
- Provides synchronized and mirrored motion patterns

**Vision Tracking** (`piper_vision_control.py`)
- Integrates MediaPipe pose estimation with robot control
- Maps human arm angles (shoulder/elbow) to robot joints
- Supports both legacy `mp.solutions` API and new Pose Landmarker model
- Key parameters: `--smoothing`, `--max-step`, `--shoulder-gain`, `--elbow-gain`
- Outputs CSV log (`vision_log.csv`) for debugging tracking behavior

**Motion Logger** (`piper_control.py`)
- Collects telemetry: target positions, actual positions, velocities, torques
- Analyzes jitter, oscillation, and position error statistics
- Saves detailed motion logs to `motion_log.txt`
- Use `logger.save()` to export analysis

### MuJoCo Models

**piper.xml**
- Single Piper X arm model in MJCF format
- Meshes loaded from `piper_ros/src/piper_description/meshes/`
- Joint hierarchy: base_link → link1 (base rotation) → link2 (shoulder) → link3 (elbow) → link4-6 (wrist) → gripper
- Position actuators with built-in PD control
- Timestep: 0.002s, gravity: -9.81 m/s²

**piper_dual.xml**
- Two Piper X arms (left/right prefixes: L_joint1-6, R_joint1-6)
- Arms positioned at (x=-0.3, x=0.3) for symmetric layout

### Control Architecture

**Position Servo Strategy**
- MuJoCo position actuators handle PD control internally
- `control_step()` sets `data.ctrl[i]` to target joint angles
- Gripper uses two opposing actuators (joint7/joint8) with negated values
- Lower kp values prevent aggressive response; higher kd adds damping to avoid oscillation

**Joint Naming Convention**
- Arm joints: `joint1` (base rotation), `joint2` (shoulder), `joint3` (elbow), `joint4-6` (wrist)
- Gripper: `joint7` (left finger), `joint8` (right finger, negated)
- Dual arms prefix: `L_` or `R_` before joint name

**Vision Mapping Pipeline**
1. MediaPipe extracts shoulder/elbow/wrist landmarks from webcam
2. Calculate 3D vectors between landmarks
3. Compute shoulder yaw/pitch and elbow flexion angles
4. Apply smoothing, gain scaling, and step limiting
5. Map to robot joint targets (configurable with `--swap-shoulder-elbow`, sign flips)
6. PiperArm applies position servo control

### File Dependencies

- `piper_ros/` must exist at repo root for mesh loading
- Missing `piper_ros/` causes "file not found" errors during model initialization
- MediaPipe Pose Landmarker model auto-downloads to `models/` if legacy API unavailable
- Logs written to: `motion_log.txt` (arm telemetry), `vision_log.csv` (tracking data)

## Platform Requirements

- macOS (tested on Apple Silicon)
- Python 3.10+
- MuJoCo 3.0+ (provides `mjpython` runtime)
- MediaPipe for pose estimation
- OpenCV for webcam capture

## Common Issues

**Arm movement is too slow or unstable**
- Tune PD gains in `piper_control.py` (kp/kd arrays around line 210)
- Lower kp reduces aggressiveness; higher kd adds damping

**Vision tracking too responsive/twitchy**
- Reduce `--smoothing` (try 0.6) and decrease `--max-step` (try 0.3)
- Lower `--base-gain`, `--shoulder-gain`, `--elbow-gain`

**Vision tracking too sluggish**
- Increase `--smoothing` (try 0.9) and increase `--max-step` (try 0.6)
- Defaults: smoothing=0.75, max-step=0.4, shoulder-gain=1.0, elbow-gain=1.1, base-gain=1.5

**Vision tracking falls back to "holding" pose**
- Use `--ignore-visibility` or lower `--min-visibility` (default: 0.3)
- MediaPipe may not detect arm landmarks with sufficient confidence

**Base rotation not following torso**
- Ensure `--base-from-yaw` is enabled (default)
- Torso rotation uses shoulder line angle in camera frame
- Adjust `--base-gain` (default: 1.5, range: 1.0-2.0)

**Mesh files not found**
- Ensure `piper_ros/` is cloned at repo root
- Check that `piper_ros/src/piper_description/meshes/` contains .STL files
