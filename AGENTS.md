# Repository Guidelines

## Project Structure & Module Organization
- `piper_control.py` and `piper_dual_control.py` are the main entry points for single-arm and dual-arm demos.
- `piper_simulation.py` is a minimal simulation runner.
- `piper.xml` and `piper_dual.xml` are the MuJoCo models.
- `piper_ros/` is expected at the repo root after cloning the upstream Piper ROS package for URDF/mesh assets.
- `motion_log.txt` is generated at runtime when logging is enabled; do not hand-edit it.

## Build, Test, and Development Commands
- Set up the environment:
  - `python3 -m venv venv`
  - `source venv/bin/activate`
  - `pip install -r requirements.txt`
- Run the single-arm demos with MuJoCo’s GUI runtime:
  - `mjpython piper_control.py --mode wave`
- Run the dual-arm demos:
  - `mjpython piper_dual_control.py --mode mirror`
- There is no separate build step; scripts run directly via `mjpython`.

## Coding Style & Naming Conventions
- Python style follows PEP 8: 4-space indentation, snake_case for functions/variables, UpperCamelCase for classes.
- Keep new files in the repo root unless they clearly belong in a new module folder.
- Keep mode flags lowercase (e.g., `--mode wave`, `--mode pick`).
- Prefer small, readable helper functions over long monolithic loops.

## Testing Guidelines
- No automated test suite is currently present.
- If you add tests, place them under a new `tests/` directory and use `pytest` with filenames like `test_*.py`.

## Commit & Pull Request Guidelines
- Existing history contains a concise, imperative commit message (`Initial commit: ...`). Keep messages short and descriptive.
- PRs should include:
  - A brief summary of behavior changes
  - The demo mode(s) exercised (e.g., `--mode pick`)
  - Screenshots or short clips if the visual motion changes

## Configuration Notes
- MuJoCo 3.0+ and macOS are assumed; Apple Silicon has been the primary target.
- Ensure `piper_ros/` is cloned at the repo root so meshes resolve correctly.
