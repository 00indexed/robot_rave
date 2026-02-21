# Vision Tracking Fix Plan

## Issue 1: Inverted/Mirrored Movement

**Problem**:
- You move your RIGHT arm, but code tracks LEFT landmarks (line 520)
- This makes movements backwards/inverted

**Root Cause**:
When `--mirror` is enabled (default), it:
1. Flips the camera feed horizontally (line 611)
2. Swaps which arm to track: `effective_side = "left" if args.side == "right" else "right"` (line 520)

This was designed for "selfie mode" where your right arm appears on the left side of the screen. But the landmark swap is backwards.

**Fix Options**:

**Option A (Quick)**: Disable mirror mode
```bash
mjpython piper_vision_control.py --no-mirror
```
This tracks your actual right arm, but camera view won't be mirrored (less intuitive)

**Option B (Better)**: Fix the mirror logic
- Remove the side swap on line 518-520
- Keep the visual flip but track the correct landmarks
- When mirror=True and side=right, should still track RIGHT landmarks (because the flip is visual only)

**Option C (Best)**: Add explicit sign flip for mirrored mode
- Keep current behavior but add `--mirror-joints` flag that negates shoulder_yaw
- This way mirror mode can work correctly

## Issue 2: Base Rotation (Joint 0) Doesn't Follow Torso Pivot

**Problem**:
- Current: `joint[0] = shoulder_yaw * 3.0` (line 268)
- `shoulder_yaw` = angle of ARM relative to TORSO
- When you pivot torso, shoulder_yaw doesn't change much (arm stays centered on torso)

**What We Need**:
- `joint[0]` should track TORSO rotation in world/camera frame
- Calculate: angle of torso X-axis (shoulder line) relative to camera frame

**Fix Implementation**:
Add new calculation:
```python
# Torso rotation relative to camera (not used currently)
torso_yaw = arctan2(x_axis[0], x_axis[2])  # Angle of shoulder line in camera frame
```

Replace line 268:
```python
if cfg.base_from_yaw:
    # Use torso rotation in camera frame, not shoulder angle relative to torso
    torso_yaw = float(np.arctan2(x_axis[0], x_axis[2]))
    base = torso_yaw * cfg.base_gain
else:
    base = (wrist.x - cfg.center_x) * cfg.base_gain
```

## Recommended Fix Order

1. **First**: Fix mirroring (Option B - correct the landmark selection)
2. **Second**: Fix base rotation to use torso_yaw instead of shoulder_yaw
3. **Test**: Verify both issues resolved
4. **Optional**: Add more DOF (wrist tracking)

## Implementation Steps

Would you like me to:
A. Implement both fixes now?
B. Just fix the mirror issue first and test?
C. Just fix the base rotation first and test?
D. Something else?
