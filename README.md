# Piper X Robot Arm Simulation

MuJoCo-based simulation of the Piper X robotic arm that runs natively on macOS.

## Features

- Native macOS support (no Docker/VM required)
- Single and dual arm configurations
- Multiple demo motions: wave, pick-and-place, mirror dance, clap
- Position servo control for smooth, stable movement

## Setup

```bash
# Clone this repo
git clone https://github.com/00indexed/robot_rave.git
cd robot_rave

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Clone the Piper ROS package (for URDF/mesh files)
git clone https://github.com/agilexrobotics/piper_ros.git
```

## Usage

### Single Arm

```bash
# Interactive simulation
mjpython piper_control.py --mode hello   # Wave hello
mjpython piper_control.py --mode wave    # Smooth wave motion
mjpython piper_control.py --mode pick    # Pick and place
mjpython piper_control.py --mode circle  # Circular motion
```

### Dual Arms

```bash
mjpython piper_dual_control.py --mode wave    # Both arms wave
mjpython piper_dual_control.py --mode sync    # Synchronized waving
mjpython piper_dual_control.py --mode mirror  # Mirror dance
mjpython piper_dual_control.py --mode clap    # Arms clap together
```

### Vision Tracking (Camera)

```bash
# Track your right arm (default) with a mirrored selfie view
mjpython piper_vision_control.py

# Track left arm without mirroring the camera feed
mjpython piper_vision_control.py --side left --no-mirror
```

Notes:
- Press ESC in the camera window to stop.
- Tune responsiveness with `--smoothing`, `--max-step`, and `--base-gain`.
- If the arm barely moves, try `--smoothing 0.7 --max-step 0.35 --shoulder-gain 1.3 --elbow-gain 1.3`.
- If tracking keeps falling back to \"holding\", try `--ignore-visibility` or `--min-visibility 0`.
- If the shoulder/elbow mapping feels swapped or inverted, try `--swap-shoulder-elbow` or `--shoulder-sign -1 --elbow-sign -1`.
- If the tracked arm feels swapped, try `--no-mirror` or flip `--side`.
- If MediaPipe’s legacy `solutions` API is unavailable, the script will download a Pose Landmarker model into `models/` (override with `--model`).
- A CSV log is written by default (`vision_log.csv`). Disable with `--no-log` or change location with `--log-path`.

## Controls

| Action | Control |
|--------|---------|
| Rotate view | Mouse drag |
| Pan view | Shift + drag |
| Zoom | Scroll |
| Pause/Resume | Space |
| Reset | Backspace |
| Exit | ESC |

## Files

- `piper.xml` - Single arm MuJoCo model
- `piper_dual.xml` - Dual arm MuJoCo model
- `piper_control.py` - Single arm controller with motion demos
- `piper_dual_control.py` - Dual arm controller
- `piper_vision_control.py` - Camera-based pose tracking controller
- `piper_simulation.py` - Basic simulation script

## Requirements

- Python 3.10+
- macOS (tested on Apple Silicon)
- MuJoCo 3.0+
- MediaPipe (for pose estimation)
- OpenCV (for webcam capture)
