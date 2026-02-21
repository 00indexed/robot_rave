#!/usr/bin/env python3
"""
Test what happens when arm is straight out horizontally.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/Henry.Ibeneme/dev/hackathon/robot_rave')

from piper_vision_control import _map_landmarks_to_joints, TrackingConfig
from dataclasses import dataclass

@dataclass
class MockLandmark:
    x: float
    y: float
    z: float
    visibility: float = 1.0

def create_realistic_pose(arm_angle_from_down: float, elbow_bend_degrees: float = 0):
    """
    Create landmarks for specific arm pose.
    arm_angle_from_down: 0° = arm hanging down, 90° = horizontal forward, 180° = up
    elbow_bend_degrees: 0° = straight, 90° = bent 90°, 150° = max bend
    """
    landmarks = [None] * 33

    # Torso center at origin, facing camera
    # Shoulders 0.2 apart horizontally
    left_shoulder = np.array([0.4, 0.4, 0.0])
    right_shoulder = np.array([0.6, 0.4, 0.0])

    # Hips below
    left_hip = np.array([0.45, 0.6, 0.0])
    right_hip = np.array([0.55, 0.6, 0.0])

    # Right arm from shoulder
    shoulder_pos = right_shoulder
    arm_length = 0.25

    # Convert angle to radians
    angle_rad = np.radians(arm_angle_from_down)

    # Upper arm direction
    # 0° = pointing down (0, 1, 0)
    # 90° = pointing forward (0, 0, 1)
    # 180° = pointing up (0, -1, 0)
    upper_arm_dir = np.array([
        0,  # No left-right component
        -np.cos(angle_rad),  # Vertical: down is positive Y
        np.sin(angle_rad),   # Forward: positive Z
    ])

    elbow_pos = shoulder_pos + upper_arm_dir * arm_length

    # Forearm
    elbow_bend_rad = np.radians(elbow_bend_degrees)

    # Rotate forearm relative to upper arm
    # Straight arm: forearm continues in same direction
    # Bent arm: forearm rotates in the pitch plane
    forearm_dir = np.array([
        0,
        -np.cos(angle_rad + elbow_bend_rad),
        np.sin(angle_rad + elbow_bend_rad),
    ])

    wrist_pos = elbow_pos + forearm_dir * arm_length

    # Create landmarks
    landmarks[11] = MockLandmark(left_shoulder[0], left_shoulder[1], left_shoulder[2])
    landmarks[12] = MockLandmark(right_shoulder[0], right_shoulder[1], right_shoulder[2])
    landmarks[23] = MockLandmark(left_hip[0], left_hip[1], left_hip[2])
    landmarks[24] = MockLandmark(right_hip[0], right_hip[1], right_hip[2])
    landmarks[14] = MockLandmark(elbow_pos[0], elbow_pos[1], elbow_pos[2])
    landmarks[16] = MockLandmark(wrist_pos[0], wrist_pos[1], wrist_pos[2])

    return landmarks

cfg = TrackingConfig(
    side="right",
    min_visibility=0.3,
    ignore_visibility=True,
    smoothing=0.75,
    max_step=0.4,
    base_gain=1.5,
    shoulder_gain=1.0,  # Updated
    elbow_gain=1.1,     # Updated
    shoulder_sign=-1.0,
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
print("REALISTIC ARM POSITION TEST")
print("="*80)
print("\nExpected behavior:")
print("  Arm hanging down (0°)      → Robot shoulder ~0° (arm down)")
print("  Arm horizontal forward (90°) → Robot shoulder ~90° (mid-range)")
print("  Arm pointing up (180°)     → Robot shoulder ~180° (arm up)")
print("\n" + "-"*80)
print(f"{'Human Arm Pose':<30} {'shoulder_pitch':>15} {'Robot Joint[1]':>15} {'Match':>10}")
print("-"*80)

test_cases = [
    ("Arm hanging down (0°)", 0),
    ("Arm 30° from down", 30),
    ("Arm 45° from down", 45),
    ("Arm 60° from down", 60),
    ("Arm horizontal (90°)", 90),
    ("Arm 120° from down", 120),
    ("Arm 135° from down", 135),
    ("Arm 150° from down", 150),
    ("Arm pointing up (180°)", 180),
]

for desc, arm_angle in test_cases:
    landmarks = create_realistic_pose(arm_angle, elbow_bend_degrees=0)

    result = _map_landmarks_to_joints(landmarks, cfg, frame_shape)
    if result:
        joints, diag = result

        shoulder_pitch_deg = np.degrees(diag.shoulder_pitch)
        robot_shoulder_deg = np.degrees(joints[1])

        # Expected robot angle (should roughly equal arm_angle for 1:1 mapping)
        expected = arm_angle
        error = abs(robot_shoulder_deg - expected)
        match = "✓" if error < 10 else f"✗ ({error:.0f}° off)"

        print(f"{desc:<30} {shoulder_pitch_deg:>15.1f}° {robot_shoulder_deg:>15.1f}° {match:>10}")

print("\n" + "="*80)
print("ELBOW BEND TEST")
print("="*80)
print(f"{'Human Arm Pose':<30} {'elbow_flex':>15} {'Robot Joint[2]':>15} {'Expected':>15}")
print("-"*80)

elbow_tests = [
    ("Straight arm (0° bend)", 90, 0),      # Arm horizontal, straight
    ("Slight bend (30°)", 90, 30),
    ("Medium bend (60°)", 90, 60),
    ("90° bend", 90, 90),
    ("Heavy bend (120°)", 90, 120),
    ("Max bend (140°)", 90, 140),
]

for desc, arm_angle, elbow_bend in elbow_tests:
    landmarks = create_realistic_pose(arm_angle, elbow_bend_degrees=elbow_bend)

    result = _map_landmarks_to_joints(landmarks, cfg, frame_shape)
    if result:
        joints, diag = result

        elbow_flex_deg = np.degrees(diag.elbow_flex)
        robot_elbow_deg = np.degrees(joints[2])

        # Robot elbow is negative (0 to -170)
        # elbow_flex should map to negative values
        expected_robot = -elbow_bend * 1.1  # With gain

        print(f"{desc:<30} {elbow_flex_deg:>15.1f}° {robot_elbow_deg:>15.1f}° {expected_robot:>15.1f}°")

print("\n" + "="*80)
print("If robot shoulder angle doesn't match human arm angle:")
print("  → Adjust --shoulder-offset (shifts the whole range)")
print("  → Adjust --shoulder-gain (scales the range)")
print("="*80)
