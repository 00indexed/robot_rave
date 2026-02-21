# Vision Tracking Fixes - Summary

## Problems Identified

1. **Sluggish/Unresponsive Movement**
   - Smoothing too low (0.2 = only 20% new motion)
   - Max step too small (0.12 rad = 6.9° per frame)
   - Only 1 sim step per camera frame
   - Gains too low

2. **Inverted/Mirrored Movement**
   - Mirror mode was double-swapping: flip video + swap landmarks
   - Right arm movements tracked wrong landmarks

3. **Base Rotation Not Following Torso**
   - `joint[0]` used `shoulder_yaw` (arm angle relative to torso)
   - Should use `torso_yaw` (torso rotation in camera frame)

4. **90° Offset in Torso Tracking**
   - When facing camera, torso_yaw = 90° instead of 0°
   - Needed to subtract π/2 offset

5. **Joint Limit Violations**
   - Shoulder joint exceeding 180° limit
   - No clamping to hardware limits

## Changes Made

### 1. Responsiveness Improvements (piper_vision_control.py)

**Line 468-479**: Increased default parameters
```python
--smoothing: 0.2 → 0.75        # 75% new motion vs 20%
--max-step: 0.12 → 0.4 rad     # 23° per frame vs 7°
--shoulder-gain: 1.0 → 1.3     # More responsive arm tracking
--elbow-gain: 1.0 → 1.3
--min-visibility: 0.5 → 0.3    # Less likely to lose tracking
```

**Line 640**: Increased simulation steps
```python
# Step simulation 10x per camera frame for faster response
for _ in range(10):
    arm.step()
```

### 2. Mirror Logic Fix (piper_vision_control.py:518-520)

**Before:**
```python
effective_side = args.side
if args.mirror:
    effective_side = "left" if args.side == "right" else "right"  # ❌ Double swap
```

**After:**
```python
# MediaPipe detects left/right anatomically, not based on screen position
effective_side = args.side  # ✅ Track the correct anatomical side
```

### 3. Torso Rotation Tracking (piper_vision_control.py:268-277)

**Before:**
```python
if cfg.base_from_yaw:
    base = shoulder_yaw * cfg.base_gain  # ❌ Arm angle, not torso rotation
```

**After:**
```python
# Calculate torso rotation in camera frame
# x_axis points from left shoulder to right shoulder
# When facing camera: x_axis = (1, 0, 0) → arctan2(1, 0) = π/2
# Subtract π/2 to center at 0 when facing camera
torso_yaw = float(np.arctan2(x_axis[0], x_axis[2]) - np.pi / 2.0)

if cfg.base_from_yaw:
    base = torso_yaw * cfg.base_gain  # ✅ Actual torso rotation
```

### 4. Base Gain Reduction (piper_vision_control.py:472)

```python
--base-gain: 3.0 → 1.5  # Reduced to keep within ±150° joint limits
```

### 5. Joint Limit Clamping (piper_vision_control.py:287-295)

**Added hardware limit clamping:**
```python
joints = np.array([
    np.clip(base, -2.618, 2.618),           # base: ±150°
    np.clip(shoulder_value, 0.0, 3.14),     # shoulder: 0° to 180°
    np.clip(elbow_value, -2.967, 0.0),      # elbow: -170° to 0°
    0.0, 0.0, 0.0,                          # wrist joints unused
])
```

### 6. Updated Diagnostics

- Added `torso_yaw` to `MappingDiagnostics` dataclass
- Updated CSV logging to include `torso_yaw` column
- Changed on-screen display to show `torso_yaw` instead of `shoulder_yaw`

## Test Results

✅ **Torso Rotation Tracking**
- Facing camera → joint[0] = 0°
- Rotate left 30° → joint[0] = 45°
- Rotate right 30° → joint[0] = -45°

✅ **Arm Independence**
- Moving arm doesn't affect joint[0] when torso is stationary

✅ **Mirror Mode**
- Right arm tracking works correctly
- No more inverted movements

✅ **Joint Limits**
- All joints stay within hardware limits
- No more 275° shoulder violations

✅ **Responsiveness**
- 3.75x faster smoothing (0.2 → 0.75)
- 3.3x larger max step (0.12 → 0.4 rad)
- 10x more sim steps per frame
- 1.3x arm gain multipliers

## How to Use

```bash
# Default (right arm, mirror mode)
mjpython piper_vision_control.py

# Left arm
mjpython piper_vision_control.py --side left

# Even more responsive (tune if default is too slow)
mjpython piper_vision_control.py --smoothing 0.9 --max-step 0.6

# Less twitchy (tune if default is too jumpy)
mjpython piper_vision_control.py --smoothing 0.6 --max-step 0.3

# Different base rotation sensitivity
mjpython piper_vision_control.py --base-gain 2.0  # More sensitive to torso rotation
mjpython piper_vision_control.py --base-gain 1.0  # Less sensitive
```

## Files Modified

- `piper_vision_control.py` - Main fixes
- `VISION_FIX_PLAN.md` - Problem analysis
- `test_vision_mapping.py` - Unit tests for mapping logic
- `test_joint_limits.py` - Joint limit validation tests
