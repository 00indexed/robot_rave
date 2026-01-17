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
- `piper_simulation.py` - Basic simulation script

## Requirements

- Python 3.10+
- macOS (tested on Apple Silicon)
- MuJoCo 3.0+
