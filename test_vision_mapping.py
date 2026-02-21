#!/usr/bin/env python3
"""
Test script to validate vision mapping logic without requiring a camera.
Simulates different body poses and checks the joint mapping.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

# Mock landmarks structure
@dataclass
class MockLandmark:
    x: float
    y: float
    z: float
    visibility: float = 1.0


def create_mock_landmarks(
    torso_rotation: float = 0.0,  # Rotation around vertical axis (radians)
    arm_forward: float = 0.0,      # How far forward the arm is extended
    arm_up: float = 0.5,           # How high the arm is raised (0-1)
    elbow_bend: float = 0.5,       # How bent the elbow is (0-1)
    side: str = "right"
):
    """Create mock MediaPipe landmarks for testing."""
    landmarks = [None] * 33

    # Torso center at origin
    torso_center = np.array([0.5, 0.5, 0.0])

    # Create rotation matrix for torso
    cos_r = np.cos(torso_rotation)
    sin_r = np.sin(torso_rotation)

    # Shoulders (0.2 units apart, rotated by torso_rotation)
    shoulder_offset = 0.1
    left_shoulder = torso_center + np.array([-shoulder_offset * cos_r, -0.1, -shoulder_offset * sin_r])
    right_shoulder = torso_center + np.array([shoulder_offset * cos_r, -0.1, shoulder_offset * sin_r])

    # Hips (0.15 units apart, below shoulders)
    hip_offset = 0.075
    left_hip = torso_center + np.array([-hip_offset * cos_r, 0.2, -hip_offset * sin_r])
    right_hip = torso_center + np.array([hip_offset * cos_r, 0.2, hip_offset * sin_r])

    # Arm joints
    if side == "right":
        shoulder = right_shoulder
    else:
        shoulder = left_shoulder

    # Elbow position (based on arm_up and arm_forward)
    arm_length = 0.15
    elbow = shoulder + np.array([
        arm_forward * arm_length,
        -arm_up * arm_length,
        0.05
    ])

    # Wrist position (based on elbow_bend)
    forearm_length = 0.15
    wrist_offset = np.array([
        arm_forward * forearm_length * elbow_bend,
        -arm_up * forearm_length * (1.0 - elbow_bend),
        0.05
    ])
    wrist = elbow + wrist_offset

    # Build landmarks array
    landmarks[11] = MockLandmark(left_shoulder[0], left_shoulder[1], left_shoulder[2])
    landmarks[12] = MockLandmark(right_shoulder[0], right_shoulder[1], right_shoulder[2])
    landmarks[23] = MockLandmark(left_hip[0], left_hip[1], left_hip[2])
    landmarks[24] = MockLandmark(right_hip[0], right_hip[1], right_hip[2])

    if side == "right":
        landmarks[12] = MockLandmark(shoulder[0], shoulder[1], shoulder[2])
        landmarks[14] = MockLandmark(elbow[0], elbow[1], elbow[2])
        landmarks[16] = MockLandmark(wrist[0], wrist[1], wrist[2])
    else:
        landmarks[11] = MockLandmark(shoulder[0], shoulder[1], shoulder[2])
        landmarks[13] = MockLandmark(elbow[0], elbow[1], elbow[2])
        landmarks[15] = MockLandmark(wrist[0], wrist[1], wrist[2])

    return landmarks


def test_torso_rotation():
    """Test that torso rotation correctly maps to joint[0]."""
    print("=" * 60)
    print("TEST 1: Torso Rotation → Joint[0] (Base)")
    print("=" * 60)

    # Import the actual mapping function
    import sys
    sys.path.insert(0, '/Users/Henry.Ibeneme/dev/hackathon/robot_rave')
    from piper_vision_control import _map_landmarks_to_joints, TrackingConfig

    cfg = TrackingConfig(
        side="right",
        min_visibility=0.3,
        ignore_visibility=True,
        smoothing=0.75,
        max_step=0.4,
        base_gain=1.5,
        shoulder_gain=1.0,
        elbow_gain=1.1,
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

    test_cases = [
        ("Center (no rotation)", 0.0),
        ("Rotate left 30°", np.radians(30)),
        ("Rotate right 30°", np.radians(-30)),
        ("Rotate left 60°", np.radians(60)),
        ("Rotate right 60°", np.radians(-60)),
    ]

    frame_shape = (480, 640, 3)

    for name, rotation in test_cases:
        landmarks = create_mock_landmarks(
            torso_rotation=rotation,
            arm_forward=0.0,
            arm_up=0.5,
            elbow_bend=0.5,
            side="right"
        )

        result = _map_landmarks_to_joints(landmarks, cfg, frame_shape)
        if result is not None:
            joints, diagnostics = result
            print(f"\n{name}:")
            print(f"  Torso rotation: {np.degrees(rotation):>6.1f}°")
            print(f"  Torso yaw:      {np.degrees(diagnostics.torso_yaw):>6.1f}°")
            print(f"  Joint[0] (base):{joints[0]:>6.2f} rad ({np.degrees(joints[0]):>6.1f}°)")
            print(f"  Joint[1] (shldr):{joints[1]:>6.2f} rad")
            print(f"  Joint[2] (elbow):{joints[2]:>6.2f} rad")
        else:
            print(f"\n{name}: FAILED (no mapping)")


def test_arm_movements():
    """Test that arm movements don't affect joint[0] when torso is stationary."""
    print("\n" + "=" * 60)
    print("TEST 2: Arm Movement (Stationary Torso) → Joint[0] Should Stay ~0")
    print("=" * 60)

    import sys
    sys.path.insert(0, '/Users/Henry.Ibeneme/dev/hackathon/robot_rave')
    from piper_vision_control import _map_landmarks_to_joints, TrackingConfig

    cfg = TrackingConfig(
        side="right",
        min_visibility=0.3,
        ignore_visibility=True,
        smoothing=0.75,
        max_step=0.4,
        base_gain=1.5,
        shoulder_gain=1.0,
        elbow_gain=1.1,
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

    test_cases = [
        ("Arm down", 0.2, 0.0),
        ("Arm mid", 0.5, 0.0),
        ("Arm up", 0.8, 0.0),
        ("Arm forward", 0.5, 0.5),
        ("Arm back", 0.5, -0.3),
    ]

    for name, arm_up, arm_forward in test_cases:
        landmarks = create_mock_landmarks(
            torso_rotation=0.0,  # Torso NOT rotated
            arm_forward=arm_forward,
            arm_up=arm_up,
            elbow_bend=0.5,
            side="right"
        )

        result = _map_landmarks_to_joints(landmarks, cfg, frame_shape)
        if result is not None:
            joints, diagnostics = result
            print(f"\n{name}:")
            print(f"  Torso yaw:      {np.degrees(diagnostics.torso_yaw):>6.1f}° (should be ~0°)")
            print(f"  Joint[0] (base):{joints[0]:>6.2f} rad (should be ~0)")
            print(f"  Joint[1] (shldr):{joints[1]:>6.2f} rad")
            print(f"  Joint[2] (elbow):{joints[2]:>6.2f} rad")


def test_mirror_mode():
    """Test that right arm tracking works correctly."""
    print("\n" + "=" * 60)
    print("TEST 3: Right Arm Tracking (Mirror Mode Fix)")
    print("=" * 60)

    import sys
    sys.path.insert(0, '/Users/Henry.Ibeneme/dev/hackathon/robot_rave')
    from piper_vision_control import _map_landmarks_to_joints, TrackingConfig

    cfg_right = TrackingConfig(
        side="right",
        min_visibility=0.3,
        ignore_visibility=True,
        smoothing=0.75,
        max_step=0.4,
        base_gain=1.5,
        shoulder_gain=1.0,
        elbow_gain=1.1,
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

    # Test right arm at different positions
    landmarks = create_mock_landmarks(
        torso_rotation=0.0,
        arm_forward=0.0,
        arm_up=0.5,
        elbow_bend=0.5,
        side="right"
    )

    result = _map_landmarks_to_joints(landmarks, cfg_right, frame_shape)
    if result is not None:
        joints, diagnostics = result
        print("\nRight arm tracking:")
        print(f"  Side: right")
        print(f"  Mapped successfully: YES")
        print(f"  Joint values: {joints[:3]}")
    else:
        print("\nRight arm tracking: FAILED")


if __name__ == "__main__":
    try:
        test_torso_rotation()
        test_arm_movements()
        test_mirror_mode()
        print("\n" + "=" * 60)
        print("TESTS COMPLETE")
        print("=" * 60)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
