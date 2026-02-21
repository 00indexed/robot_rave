#!/usr/bin/env python3
"""
Comprehensive vision tracking test - simulates realistic tracking scenarios
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/Henry.Ibeneme/dev/hackathon/robot_rave')

from test_vision_mapping import create_mock_landmarks
from piper_vision_control import _map_landmarks_to_joints, TrackingConfig, _smooth_joints

def test_scenario(name, description, landmarks_sequence, cfg):
    """Test a sequence of poses."""
    print(f"\n{'='*70}")
    print(f"SCENARIO: {name}")
    print(f"{'='*70}")
    print(f"Description: {description}\n")

    frame_shape = (480, 640, 3)
    last_joints = None

    for i, (pose_name, landmarks) in enumerate(landmarks_sequence):
        result = _map_landmarks_to_joints(landmarks, cfg, frame_shape)

        if result is not None:
            joints, diagnostics = result

            # Apply smoothing like the real system
            smoothed = _smooth_joints(last_joints, joints, cfg)
            last_joints = smoothed

            # Check joint limits
            violations = []
            limits = [(-2.618, 2.618), (0, 3.14), (-2.967, 0),
                     (-1.745, 1.745), (-1.22, 1.22), (-2.094, 2.094)]

            for j in range(6):
                if not (limits[j][0] <= smoothed[j] <= limits[j][1]):
                    violations.append(j)

            status = "✓" if not violations else f"✗ LIMIT VIOLATION: joints {violations}"

            print(f"Frame {i+1}: {pose_name}")
            print(f"  Torso yaw:  {np.degrees(diagnostics.torso_yaw):>6.1f}°")
            print(f"  Joint[0]:   {smoothed[0]:>6.2f} rad ({np.degrees(smoothed[0]):>6.1f}°) - Base rotation")
            print(f"  Joint[1]:   {smoothed[1]:>6.2f} rad ({np.degrees(smoothed[1]):>6.1f}°) - Shoulder")
            print(f"  Joint[2]:   {smoothed[2]:>6.2f} rad ({np.degrees(smoothed[2]):>6.1f}°) - Elbow")
            print(f"  Status:     {status}\n")
        else:
            print(f"Frame {i+1}: {pose_name} - FAILED TO MAP\n")


def main():
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

    # SCENARIO 1: User waves hello (no torso rotation)
    wave_sequence = [
        ("Start: Arm at side", create_mock_landmarks(0.0, 0.0, 0.3, 0.5, "right")),
        ("Raise arm mid", create_mock_landmarks(0.0, 0.0, 0.5, 0.5, "right")),
        ("Raise arm high", create_mock_landmarks(0.0, 0.0, 0.7, 0.5, "right")),
        ("Lower to mid", create_mock_landmarks(0.0, 0.0, 0.5, 0.5, "right")),
        ("Back to side", create_mock_landmarks(0.0, 0.0, 0.3, 0.5, "right")),
    ]
    test_scenario(
        "Wave Hello",
        "User raises and lowers arm without moving torso. Base should stay at ~0°",
        wave_sequence,
        cfg
    )

    # SCENARIO 2: User pivots torso left and right
    torso_sequence = [
        ("Center", create_mock_landmarks(0.0, 0.0, 0.5, 0.5, "right")),
        ("Pivot left 15°", create_mock_landmarks(np.radians(15), 0.0, 0.5, 0.5, "right")),
        ("Pivot left 30°", create_mock_landmarks(np.radians(30), 0.0, 0.5, 0.5, "right")),
        ("Back to center", create_mock_landmarks(0.0, 0.0, 0.5, 0.5, "right")),
        ("Pivot right 15°", create_mock_landmarks(np.radians(-15), 0.0, 0.5, 0.5, "right")),
        ("Pivot right 30°", create_mock_landmarks(np.radians(-30), 0.0, 0.5, 0.5, "right")),
        ("Back to center", create_mock_landmarks(0.0, 0.0, 0.5, 0.5, "right")),
    ]
    test_scenario(
        "Torso Rotation",
        "User pivots torso while keeping arm in same position. Base should follow torso rotation.",
        torso_sequence,
        cfg
    )

    # SCENARIO 3: Bend and straighten elbow
    elbow_sequence = [
        ("Straight arm", create_mock_landmarks(0.0, 0.0, 0.5, 0.1, "right")),
        ("Slight bend", create_mock_landmarks(0.0, 0.0, 0.5, 0.3, "right")),
        ("Medium bend", create_mock_landmarks(0.0, 0.0, 0.5, 0.5, "right")),
        ("Heavy bend", create_mock_landmarks(0.0, 0.0, 0.5, 0.8, "right")),
        ("Back straight", create_mock_landmarks(0.0, 0.0, 0.5, 0.1, "right")),
    ]
    test_scenario(
        "Elbow Flexion",
        "User bends and straightens elbow. Joint[2] should change, Joint[0] stays ~0°",
        elbow_sequence,
        cfg
    )

    # SCENARIO 4: Reach forward
    reach_sequence = [
        ("Arm neutral", create_mock_landmarks(0.0, 0.0, 0.5, 0.5, "right")),
        ("Reach slight forward", create_mock_landmarks(0.0, 0.2, 0.5, 0.5, "right")),
        ("Reach more forward", create_mock_landmarks(0.0, 0.4, 0.5, 0.5, "right")),
        ("Pull back", create_mock_landmarks(0.0, 0.0, 0.5, 0.5, "right")),
    ]
    test_scenario(
        "Forward Reach",
        "User reaches arm forward and back. Shoulder pitch should change.",
        reach_sequence,
        cfg
    )

    # SCENARIO 5: Complex motion - torso + arm
    complex_sequence = [
        ("Start center", create_mock_landmarks(0.0, 0.0, 0.5, 0.5, "right")),
        ("Torso left + arm up", create_mock_landmarks(np.radians(20), 0.0, 0.7, 0.5, "right")),
        ("Torso left + arm forward", create_mock_landmarks(np.radians(20), 0.3, 0.6, 0.4, "right")),
        ("Torso right + arm down", create_mock_landmarks(np.radians(-20), 0.0, 0.3, 0.6, "right")),
        ("Back center", create_mock_landmarks(0.0, 0.0, 0.5, 0.5, "right")),
    ]
    test_scenario(
        "Complex Motion",
        "Combined torso rotation and arm movement. All joints should respond correctly.",
        complex_sequence,
        cfg
    )

    # SCENARIO 6: Stress test - rapid movements
    print(f"\n{'='*70}")
    print(f"SCENARIO: Rapid Movement Stress Test")
    print(f"{'='*70}")
    print(f"Testing smoothing and max_step limiting with rapid pose changes\n")

    frame_shape = (480, 640, 3)
    last_joints = None

    # Rapid alternating poses
    rapid_poses = [
        create_mock_landmarks(0.0, 0.0, 0.3, 0.5, "right"),
        create_mock_landmarks(np.radians(45), 0.0, 0.8, 0.2, "right"),
        create_mock_landmarks(0.0, 0.0, 0.3, 0.5, "right"),
        create_mock_landmarks(np.radians(-45), 0.0, 0.8, 0.2, "right"),
    ]

    max_deltas = []
    for i, landmarks in enumerate(rapid_poses * 3):  # Repeat 3 times
        result = _map_landmarks_to_joints(landmarks, cfg, frame_shape)
        if result:
            joints, diag = result
            smoothed = _smooth_joints(last_joints, joints, cfg)

            if last_joints is not None:
                delta = np.abs(smoothed - last_joints)
                max_delta = np.max(delta)
                max_deltas.append(max_delta)

                if i % 4 == 0:
                    print(f"Cycle {i//4 + 1}:")
                print(f"  Frame {i+1}: max joint delta = {max_delta:.3f} rad ({np.degrees(max_delta):.1f}°)")

            last_joints = smoothed

    if max_deltas:
        print(f"\nMax delta observed: {max(max_deltas):.3f} rad ({np.degrees(max(max_deltas)):.1f}°)")
        print(f"Configured max_step: {cfg.max_step:.3f} rad ({np.degrees(cfg.max_step):.1f}°)")
        if max(max_deltas) <= cfg.max_step:
            print("✓ All movements respect max_step limit")
        else:
            print("✗ Some movements exceeded max_step limit!")

    print(f"\n{'='*70}")
    print("ALL TESTS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
