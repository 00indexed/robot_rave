#!/usr/bin/env python3
"""
Diagnose the actual range of motion and mapping issues.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/Henry.Ibeneme/dev/hackathon/robot_rave')

from test_vision_mapping import create_mock_landmarks
from piper_vision_control import _map_landmarks_to_joints, TrackingConfig

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

print("="*80)
print("RANGE OF MOTION DIAGNOSTIC")
print("="*80)
print("\nTesting different arm positions to see actual angle mapping...\n")

# Test different arm heights
print("-"*80)
print("ARM HEIGHT VARIATION (arm_up: 0.0 to 1.0)")
print("-"*80)
print(f"{'Human Pose':<25} {'shoulder_pitch':>15} {'Joint[1] Raw':>15} {'Joint[1] Final':>15} {'Degrees':>10}")
print("-"*80)

for arm_up in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
    landmarks = create_mock_landmarks(
        torso_rotation=0.0,
        arm_forward=0.0,
        arm_up=arm_up,
        elbow_bend=0.5,
        side="right"
    )

    result = _map_landmarks_to_joints(landmarks, cfg, frame_shape)
    if result:
        joints, diag = result

        # Calculate what the raw value would be without gain
        shoulder_pitch = diag.shoulder_pitch
        raw_value = shoulder_pitch + np.pi/2.0

        pose_desc = f"Arm up={arm_up:.1f}"
        print(f"{pose_desc:<25} {np.degrees(shoulder_pitch):>15.1f}° {np.degrees(raw_value):>15.1f}° {np.degrees(joints[1]):>15.1f}° {joints[1]:>10.2f}r")

# Joint limits for reference
print("\n" + "="*80)
print("ROBOT JOINT LIMITS")
print("="*80)
print(f"Joint[1] (Shoulder): 0° to 180° (0.00 to 3.14 rad)")
print(f"Current range usage: Check if we're using the full 0-180° range above")

# Test forward reach
print("\n" + "-"*80)
print("FORWARD REACH VARIATION (arm_forward: -0.3 to 0.5)")
print("-"*80)
print(f"{'Human Pose':<25} {'shoulder_pitch':>15} {'Joint[1] Raw':>15} {'Joint[1] Final':>15} {'Degrees':>10}")
print("-"*80)

for arm_forward in [-0.3, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    landmarks = create_mock_landmarks(
        torso_rotation=0.0,
        arm_forward=arm_forward,
        arm_up=0.5,
        elbow_bend=0.5,
        side="right"
    )

    result = _map_landmarks_to_joints(landmarks, cfg, frame_shape)
    if result:
        joints, diag = result
        shoulder_pitch = diag.shoulder_pitch
        raw_value = shoulder_pitch + np.pi/2.0

        pose_desc = f"Arm forward={arm_forward:.1f}"
        print(f"{pose_desc:<25} {np.degrees(shoulder_pitch):>15.1f}° {np.degrees(raw_value):>15.1f}° {np.degrees(joints[1]):>15.1f}° {joints[1]:>10.2f}r")

# Elbow range
print("\n" + "-"*80)
print("ELBOW BEND VARIATION (elbow_bend: 0.0 to 1.0)")
print("-"*80)
print(f"{'Human Pose':<25} {'elbow_flex':>15} {'Joint[2] Raw':>15} {'Joint[2] Final':>15} {'Degrees':>10}")
print("-"*80)

for elbow_bend in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    landmarks = create_mock_landmarks(
        torso_rotation=0.0,
        arm_forward=0.0,
        arm_up=0.5,
        elbow_bend=elbow_bend,
        side="right"
    )

    result = _map_landmarks_to_joints(landmarks, cfg, frame_shape)
    if result:
        joints, diag = result
        elbow_flex = diag.elbow_flex
        raw_value = -elbow_flex

        pose_desc = f"Elbow bend={elbow_bend:.1f}"
        print(f"{pose_desc:<25} {np.degrees(elbow_flex):>15.1f}° {np.degrees(raw_value):>15.1f}° {np.degrees(joints[2]):>15.1f}° {joints[2]:>10.2f}r")

print("\n" + "="*80)
print("ROBOT JOINT LIMITS")
print("="*80)
print(f"Joint[2] (Elbow): -170° to 0° (-2.97 to 0.00 rad)")
print(f"Current range usage: Check if we're using the full -170° to 0° range above")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("\nIf Joint[1] values cluster around 150-180° instead of using 0-180°:")
print("  → shoulder_offset needs adjustment (try negative offset)")
print("  → OR shoulder_gain needs increase")
print("\nIf Joint[2] values only use -5° to -50° instead of -170° to 0°:")
print("  → elbow_gain needs increase")
print("  → OR elbow calculation has wrong offset")
