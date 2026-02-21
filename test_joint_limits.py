#!/usr/bin/env python3
"""Test that joint outputs are within valid ranges."""

import numpy as np
import sys
sys.path.insert(0, '/Users/Henry.Ibeneme/dev/hackathon/robot_rave')

from test_vision_mapping import create_mock_landmarks
from piper_vision_control import _map_landmarks_to_joints, TrackingConfig

# Joint limits from piper_control.py
JOINT_LIMITS = [
    (-2.618, 2.618),   # joint1 (base): ±150°
    (0, 3.14),         # joint2 (shoulder): 0° to 180°
    (-2.967, 0),       # joint3 (elbow): -170° to 0°
    (-1.745, 1.745),   # joint4 (wrist1): ±100°
    (-1.22, 1.22),     # joint5 (wrist2): ±70°
    (-2.094, 2.094),   # joint6 (wrist3): ±120°
]

cfg = TrackingConfig(
    side="right",
    min_visibility=0.3,
    ignore_visibility=True,
    smoothing=0.75,
    max_step=0.4,
    base_gain=1.5,
    shoulder_gain=1.3,
    elbow_gain=1.3,
    shoulder_sign=1.0,
    elbow_sign=1.0,
    swap_shoulder_elbow=False,
    depth_scale=1.0,
    base_from_yaw=True,
    center_x=0.5,
    shoulder_offset=0.0,
    elbow_offset=0.0,
    lost_action="hold",
)

frame_shape = (480, 640, 3)

print("=" * 70)
print("JOINT LIMIT VALIDATION")
print("=" * 70)

test_cases = [
    ("Arm down", 0.2, 0.0, 0.5, 0.0),
    ("Arm mid-height", 0.5, 0.0, 0.5, 0.0),
    ("Arm up", 0.8, 0.0, 0.5, 0.0),
    ("Arm forward", 0.5, 0.5, 0.5, 0.0),
    ("Straight arm", 0.5, 0.0, 0.1, 0.0),
    ("Bent arm", 0.5, 0.0, 0.9, 0.0),
    ("Torso left", 0.5, 0.0, 0.5, np.radians(30)),
    ("Torso right", 0.5, 0.0, 0.5, np.radians(-30)),
]

violations = []

for name, arm_up, arm_forward, elbow_bend, torso_rot in test_cases:
    landmarks = create_mock_landmarks(
        torso_rotation=torso_rot,
        arm_forward=arm_forward,
        arm_up=arm_up,
        elbow_bend=elbow_bend,
        side="right"
    )

    result = _map_landmarks_to_joints(landmarks, cfg, frame_shape)
    if result is not None:
        joints, diagnostics = result

        print(f"\n{name}:")
        has_violation = False
        for i in range(6):
            min_limit, max_limit = JOINT_LIMITS[i]
            value = joints[i]
            in_range = min_limit <= value <= max_limit
            status = "✓" if in_range else "✗ VIOLATION"

            if not in_range:
                has_violation = True
                violations.append((name, i, value, min_limit, max_limit))

            print(f"  joint[{i}]: {value:>6.2f} rad ({np.degrees(value):>7.1f}°) "
                  f"[{min_limit:>5.2f}, {max_limit:>5.2f}] {status}")

print("\n" + "=" * 70)
if violations:
    print(f"VIOLATIONS FOUND: {len(violations)}")
    print("=" * 70)
    for name, joint_idx, value, min_lim, max_lim in violations:
        print(f"  {name}: joint[{joint_idx}] = {value:.2f} rad (limit: [{min_lim:.2f}, {max_lim:.2f}])")
else:
    print("ALL JOINTS WITHIN LIMITS ✓")
    print("=" * 70)
