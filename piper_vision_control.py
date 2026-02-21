#!/usr/bin/env python3
"""
Camera-based vision tracking for the Piper X arm.

Usage:
  mjpython piper_vision_control.py
  mjpython piper_vision_control.py --side left --no-mirror
"""

import argparse
import csv
import shutil
import ssl
import subprocess
import time
import urllib.request
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
import mujoco.viewer

from piper_control import PiperArm


HAS_SOLUTIONS = hasattr(mp, "solutions")
if HAS_SOLUTIONS:
    POSE_LANDMARKS = mp.solutions.pose.PoseLandmark
    POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
    DRAW_UTILS = mp.solutions.drawing_utils
    DRAW_STYLES = mp.solutions.drawing_styles
else:
    POSE_LANDMARKS = None
    POSE_CONNECTIONS = None
    DRAW_UTILS = None
    DRAW_STYLES = None

LEFT_SHOULDER_IDX = 11
LEFT_ELBOW_IDX = 13
LEFT_WRIST_IDX = 15
RIGHT_SHOULDER_IDX = 12
RIGHT_ELBOW_IDX = 14
RIGHT_WRIST_IDX = 16
LEFT_HIP_IDX = 23
RIGHT_HIP_IDX = 24

DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_lite.task"


@dataclass
class TrackingConfig:
    side: str
    min_visibility: float
    ignore_visibility: bool
    smoothing: float
    max_step: float
    base_gain: float
    shoulder_gain: float
    elbow_gain: float
    shoulder_sign: float
    elbow_sign: float
    swap_shoulder_elbow: bool
    depth_scale: float
    base_from_yaw: bool
    center_x: float
    shoulder_offset: float
    elbow_offset: float
    lost_action: str


@dataclass
class MappingDiagnostics:
    shoulder_yaw: float
    shoulder_pitch: float
    elbow_flex: float
    torso_yaw: float
    torso_ok: bool


def _angle_between(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Return the angle between two vectors in radians."""
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-8
    if denom <= 1e-8:
        return 0.0
    cos_angle = float(np.clip(np.dot(vec_a, vec_b) / denom, -1.0, 1.0))
    return float(np.arccos(cos_angle))


def _normalize(vec: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return None
    return vec / norm


def _landmark_ok(lm, min_visibility: float, ignore_visibility: bool) -> bool:
    if ignore_visibility or min_visibility <= 0.0:
        return True
    visibility = getattr(lm, "visibility", 1.0)
    return visibility >= min_visibility


def _extract_arm_landmarks(landmarks, side: str, min_visibility: float, ignore_visibility: bool):
    if side == "left":
        if POSE_LANDMARKS:
            idxs = (
                POSE_LANDMARKS.LEFT_SHOULDER.value,
                POSE_LANDMARKS.LEFT_ELBOW.value,
                POSE_LANDMARKS.LEFT_WRIST.value,
            )
        else:
            idxs = (LEFT_SHOULDER_IDX, LEFT_ELBOW_IDX, LEFT_WRIST_IDX)
    else:
        if POSE_LANDMARKS:
            idxs = (
                POSE_LANDMARKS.RIGHT_SHOULDER.value,
                POSE_LANDMARKS.RIGHT_ELBOW.value,
                POSE_LANDMARKS.RIGHT_WRIST.value,
            )
        else:
            idxs = (RIGHT_SHOULDER_IDX, RIGHT_ELBOW_IDX, RIGHT_WRIST_IDX)

    shoulder = landmarks[idxs[0]]
    elbow = landmarks[idxs[1]]
    wrist = landmarks[idxs[2]]

    if not (
        _landmark_ok(shoulder, min_visibility, ignore_visibility)
        and _landmark_ok(elbow, min_visibility, ignore_visibility)
        and _landmark_ok(wrist, min_visibility, ignore_visibility)
    ):
        return None

    return shoulder, elbow, wrist


def _landmark_vec(lm, aspect: float, depth_scale: float) -> np.ndarray:
    return np.array(
        [
            float(lm.x),
            -float(lm.y) * aspect,
            -float(getattr(lm, "z", 0.0)) * depth_scale,
        ],
        dtype=float,
    )


def _map_landmarks_to_joints(
    landmarks,
    cfg: TrackingConfig,
    frame_shape: tuple[int, int, int],
) -> Optional[tuple[np.ndarray, MappingDiagnostics]]:
    extracted = _extract_arm_landmarks(
        landmarks,
        cfg.side,
        cfg.min_visibility,
        cfg.ignore_visibility,
    )
    if extracted is None:
        return None

    h, w = frame_shape[:2]
    aspect = h / max(w, 1)

    shoulder, elbow, wrist = extracted

    left_shoulder = landmarks[LEFT_SHOULDER_IDX]
    right_shoulder = landmarks[RIGHT_SHOULDER_IDX]
    left_hip = landmarks[LEFT_HIP_IDX]
    right_hip = landmarks[RIGHT_HIP_IDX]

    torso_ok = (
        _landmark_ok(left_shoulder, cfg.min_visibility, cfg.ignore_visibility)
        and _landmark_ok(right_shoulder, cfg.min_visibility, cfg.ignore_visibility)
    )
    hips_ok = (
        _landmark_ok(left_hip, cfg.min_visibility, cfg.ignore_visibility)
        and _landmark_ok(right_hip, cfg.min_visibility, cfg.ignore_visibility)
    )

    shoulder_vec = _landmark_vec(shoulder, aspect, cfg.depth_scale)
    elbow_vec = _landmark_vec(elbow, aspect, cfg.depth_scale)
    wrist_vec = _landmark_vec(wrist, aspect, cfg.depth_scale)

    if torso_ok:
        ls_vec = _landmark_vec(left_shoulder, aspect, cfg.depth_scale)
        rs_vec = _landmark_vec(right_shoulder, aspect, cfg.depth_scale)
        x_axis = _normalize(rs_vec - ls_vec)
    else:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)

    if hips_ok and torso_ok:
        lh_vec = _landmark_vec(left_hip, aspect, cfg.depth_scale)
        rh_vec = _landmark_vec(right_hip, aspect, cfg.depth_scale)
        shoulder_mid = (ls_vec + rs_vec) * 0.5
        hip_mid = (lh_vec + rh_vec) * 0.5
        y_axis = _normalize(shoulder_mid - hip_mid)
    else:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=float)

    if x_axis is None:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        torso_ok = False
    if y_axis is None:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        torso_ok = False

    z_axis = _normalize(np.cross(x_axis, y_axis))
    if z_axis is None:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        torso_ok = False

    if float(np.dot(z_axis, np.array([0.0, 0.0, 1.0]))) < 0.0:
        z_axis = -z_axis
    x_axis = _normalize(np.cross(y_axis, z_axis))
    if x_axis is None:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        torso_ok = False

    upper = elbow_vec - shoulder_vec
    forearm = wrist_vec - elbow_vec

    if np.linalg.norm(upper) < 1e-4 or np.linalg.norm(forearm) < 1e-4:
        return None

    upper_b = np.array(
        [
            float(np.dot(upper, x_axis)),
            float(np.dot(upper, y_axis)),
            float(np.dot(upper, z_axis)),
        ],
        dtype=float,
    )
    forearm_b = np.array(
        [
            float(np.dot(forearm, x_axis)),
            float(np.dot(forearm, y_axis)),
            float(np.dot(forearm, z_axis)),
        ],
        dtype=float,
    )

    shoulder_yaw = float(np.arctan2(upper_b[0], upper_b[2]))
    cos_y = float(np.cos(-shoulder_yaw))
    sin_y = float(np.sin(-shoulder_yaw))
    upper_yawed = np.array(
        [
            upper_b[0] * cos_y - upper_b[2] * sin_y,
            upper_b[1],
            upper_b[0] * sin_y + upper_b[2] * cos_y,
        ],
        dtype=float,
    )
    shoulder_pitch = float(np.arctan2(upper_yawed[1], upper_yawed[2]))

    elbow_angle = _angle_between(-upper_b, forearm_b)
    elbow_flex = np.clip(np.pi - elbow_angle, 0.0, np.pi)

    # Calculate torso rotation in camera frame
    # x_axis points from left shoulder to right shoulder
    # When facing camera: x_axis = (1, 0, 0) → arctan2(1, 0) = π/2
    # Subtract π/2 to center at 0 when facing camera
    torso_yaw = float(np.arctan2(x_axis[0], x_axis[2]) - np.pi / 2.0)

    if cfg.base_from_yaw:
        # Use torso rotation (pivot) not arm angle
        # Note: torso_yaw is already in radians, gain should be ~1.0 for direct mapping
        base = torso_yaw * cfg.base_gain
    else:
        base = (wrist.x - cfg.center_x) * cfg.base_gain

    # Apply sign to angle first, then add offset, then scale
    shoulder_value = ((shoulder_pitch * cfg.shoulder_sign + (np.pi / 2.0)) * cfg.shoulder_gain) + cfg.shoulder_offset
    elbow_value = ((-elbow_flex) * cfg.elbow_gain * cfg.elbow_sign) + cfg.elbow_offset

    if cfg.swap_shoulder_elbow:
        shoulder_value, elbow_value = elbow_value, shoulder_value

    # Clamp joints to hardware limits
    # Joint limits from piper_control.py JointLimits class
    joints = np.array([
        np.clip(base, -2.618, 2.618),           # base: ±150°
        np.clip(shoulder_value, 0.0, 3.14),     # shoulder: 0° to 180°
        np.clip(elbow_value, -2.967, 0.0),      # elbow: -170° to 0°
        0.0,                                     # wrist joints unused
        0.0,
        0.0,
    ])

    diagnostics = MappingDiagnostics(
        shoulder_yaw=shoulder_yaw,
        shoulder_pitch=shoulder_pitch,
        elbow_flex=elbow_flex,
        torso_yaw=torso_yaw,
        torso_ok=torso_ok and hips_ok,
    )
    return joints, diagnostics


def _smooth_joints(prev: Optional[np.ndarray], target: np.ndarray, cfg: TrackingConfig) -> np.ndarray:
    if prev is None:
        return target

    blended = prev * (1.0 - cfg.smoothing) + target * cfg.smoothing

    if cfg.max_step > 0.0:
        delta = np.clip(blended - prev, -cfg.max_step, cfg.max_step)
        blended = prev + delta

    return blended


def _draw_status(frame, text: str, y: int = 24, color: tuple[int, int, int] = (0, 255, 0)):
    cv2.putText(
        frame,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_arm_landmarks(frame, landmarks, side: str):
    if landmarks is None:
        return
    h, w = frame.shape[:2]
    if side == "left":
        idxs = (LEFT_SHOULDER_IDX, LEFT_ELBOW_IDX, LEFT_WRIST_IDX)
    else:
        idxs = (RIGHT_SHOULDER_IDX, RIGHT_ELBOW_IDX, RIGHT_WRIST_IDX)

    pts = []
    for idx in idxs:
        lm = landmarks[idx]
        x = int(np.clip(lm.x, 0.0, 1.0) * w)
        y = int(np.clip(lm.y, 0.0, 1.0) * h)
        pts.append((x, y))

    cv2.circle(frame, pts[0], 6, (255, 100, 0), -1)   # shoulder
    cv2.circle(frame, pts[1], 6, (0, 200, 255), -1)   # elbow
    cv2.circle(frame, pts[2], 6, (0, 255, 0), -1)     # wrist
    cv2.line(frame, pts[0], pts[1], (255, 255, 255), 2)
    cv2.line(frame, pts[1], pts[2], (255, 255, 255), 2)


def _landmark_visibility(landmarks, idx: int) -> float:
    if landmarks is None:
        return float("nan")
    lm = landmarks[idx]
    return float(getattr(lm, "visibility", float("nan")))




def _ensure_model_file(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe model to {path} ...")
    try:
        context = ssl.create_default_context()
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass

        with urllib.request.urlopen(DEFAULT_MODEL_URL, context=context) as response:
            with open(path, "wb") as output:
                shutil.copyfileobj(response, output)
    except Exception as exc:
        print(f"Download via urllib failed: {exc}")
        print("Trying curl fallback...")
        try:
            subprocess.run(
                ["/usr/bin/curl", "-L", "-o", str(path), DEFAULT_MODEL_URL],
                check=True,
            )
        except Exception as curl_exc:
            raise RuntimeError(
                "Unable to download the MediaPipe model. "
                "Download it manually and pass --model /path/to/pose_landmarker.task"
            ) from curl_exc
    return path


class PoseEstimator:
    def __init__(self, args: argparse.Namespace):
        self.mode = "solutions" if HAS_SOLUTIONS else "tasks"
        self._pose = None

        if self.mode == "solutions":
            self._pose = mp.solutions.pose.Pose(
                min_detection_confidence=args.min_detection,
                min_tracking_confidence=args.min_tracking,
            )
        else:
            try:
                from mediapipe.tasks import python  # type: ignore
                from mediapipe.tasks.python import vision  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "MediaPipe tasks API is not available. "
                    "Install a full MediaPipe build or use a supported Python version."
                ) from exc

            model_path = Path(args.model or DEFAULT_MODEL_PATH)
            model_path = _ensure_model_file(model_path)
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=args.min_detection,
                min_pose_presence_confidence=args.min_detection,
                min_tracking_confidence=args.min_tracking,
            )
            self._pose = vision.PoseLandmarker.create_from_options(options)
            if not hasattr(mp, "Image") or not hasattr(mp, "ImageFormat"):
                raise RuntimeError(
                    "MediaPipe Image classes are unavailable. "
                    "Try a different MediaPipe build or Python version."
                )
            self._image_cls = mp.Image
            self._image_format = mp.ImageFormat.SRGB

    def detect(self, rgb: np.ndarray, timestamp_ms: int):
        if self.mode == "solutions":
            results = self._pose.process(rgb)
            landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
            return landmarks, results

        image = self._image_cls(
            image_format=self._image_format,
            data=np.ascontiguousarray(rgb),
        )
        result = self._pose.detect_for_video(image, timestamp_ms)
        if result.pose_landmarks:
            return result.pose_landmarks[0], None
        return None, None

    def close(self):
        if self._pose is not None:
            self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track a human arm with the Piper X simulation.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0).")
    parser.add_argument("--side", choices=["left", "right"], default="right", help="Arm to track.")
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True,
                        help="Mirror camera feed for a selfie view (swaps left/right tracking).")
    parser.add_argument("--min-detection", type=float, default=0.5,
                        help="Min pose detection confidence.")
    parser.add_argument("--min-tracking", type=float, default=0.5,
                        help="Min pose tracking confidence.")
    parser.add_argument("--min-visibility", type=float, default=0.3,
                        help="Min landmark visibility to update targets.")
    parser.add_argument("--ignore-visibility", action="store_true",
                        help="Ignore landmark visibility gating.")
    parser.add_argument("--smoothing", type=float, default=0.75,
                        help="EMA smoothing factor (0-1).")
    parser.add_argument("--max-step", type=float, default=0.4,
                        help="Max joint delta per frame (radians).")
    parser.add_argument("--base-gain", type=float, default=1.5,
                        help="Gain for base rotation (yaw or wrist X, depending on --base-from-yaw).")
    parser.add_argument("--base-from-yaw", action=argparse.BooleanOptionalAction, default=True,
                        help="Drive base using shoulder yaw (default: enabled).")
    parser.add_argument("--shoulder-gain", type=float, default=1.0,
                        help="Scale shoulder angle response.")
    parser.add_argument("--elbow-gain", type=float, default=1.1,
                        help="Scale elbow angle response.")
    parser.add_argument("--shoulder-sign", type=float, default=-1.0,
                        help="Flip shoulder direction (use -1 to invert).")
    parser.add_argument("--elbow-sign", type=float, default=1.0,
                        help="Flip elbow direction (use -1 to invert).")
    parser.add_argument("--swap-shoulder-elbow", action="store_true",
                        help="Swap shoulder and elbow mapping.")
    parser.add_argument("--depth-scale", type=float, default=1.0,
                        help="Scale relative depth (z) before joint mapping.")
    parser.add_argument("--center-x", type=float, default=0.5,
                        help="Normalized center X for base rotation when using wrist X.")
    parser.add_argument("--shoulder-offset", type=float, default=0.0,
                        help="Offset added to shoulder joint (radians).")
    parser.add_argument("--elbow-offset", type=float, default=0.0,
                        help="Offset added to elbow joint (radians).")
    parser.add_argument("--lost-action", choices=["hold", "home"], default="hold",
                        help="What to do when pose is lost.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH),
                        help="Path to pose_landmarker.task (used if MediaPipe solutions are unavailable).")
    parser.add_argument("--preview-only", action="store_true",
                        help="Show pose preview without controlling the robot.")
    parser.add_argument("--debug", action="store_true",
                        help="Print pose and joint target diagnostics.")
    parser.add_argument("--print-every", type=int, default=30,
                        help="Print debug output every N frames (default: 30).")
    parser.add_argument("--log", action=argparse.BooleanOptionalAction, default=True,
                        help="Write detection + joint data to a CSV log (default: enabled).")
    parser.add_argument("--log-path", type=str, default="vision_log.csv",
                        help="Path to CSV log file.")
    parser.add_argument("--log-every", type=int, default=5,
                        help="Write a log row every N frames (default: 5).")
    parser.add_argument("--no-gui", action="store_true",
                        help="Disable the OpenCV preview window.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # MediaPipe detects left/right anatomically, not based on screen position
    # So we track the arm the user specified, regardless of mirror
    effective_side = args.side

    cfg = TrackingConfig(
        side=effective_side,
        min_visibility=args.min_visibility,
        ignore_visibility=args.ignore_visibility,
        smoothing=np.clip(args.smoothing, 0.0, 1.0),
        max_step=max(args.max_step, 0.0),
        base_gain=args.base_gain,
        shoulder_gain=args.shoulder_gain,
        elbow_gain=args.elbow_gain,
        shoulder_sign=args.shoulder_sign,
        elbow_sign=args.elbow_sign,
        swap_shoulder_elbow=args.swap_shoulder_elbow,
        depth_scale=args.depth_scale,
        base_from_yaw=args.base_from_yaw,
        center_x=args.center_x,
        shoulder_offset=args.shoulder_offset,
        elbow_offset=args.elbow_offset,
        lost_action=args.lost_action,
    )

    arm = PiperArm(enable_logging=False)
    home_joints = np.array([0.0, 1.0, -1.5, 0.0, 0.0, 0.0])
    arm.move_to_joint_positions(home_joints.tolist())

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera}.")

    gui_enabled = not args.no_gui
    if gui_enabled:
        try:
            cv2.namedWindow("Piper Vision Control", cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            print(f"OpenCV GUI unavailable ({exc}). Continuing without preview window.")
            gui_enabled = False

    last_joints: Optional[np.ndarray] = None
    frame_idx = 0
    log_writer = None
    log_file = None

    if args.log:
        log_file = open(args.log_path, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(
            [
                "frame",
                "time_s",
                "status",
                "wrist_x",
                "wrist_y",
                "wrist_z",
                "shoulder_vis",
                "elbow_vis",
                "wrist_vis",
                "shoulder_yaw",
                "shoulder_pitch",
                "elbow_flex",
                "torso_yaw",
                "torso_ok",
                "target_j1",
                "target_j2",
                "target_j3",
                "target_j4",
                "target_j5",
                "target_j6",
                "actual_j1",
                "actual_j2",
                "actual_j3",
                "actual_j4",
                "actual_j5",
                "actual_j6",
            ]
        )

    with PoseEstimator(args) as pose:
        try:
            viewer_ctx = (
                mujoco.viewer.launch_passive(arm.model, arm.data)
                if not args.preview_only
                else None
            )
            with viewer_ctx if viewer_ctx is not None else nullcontext() as viewer:
                while (viewer is None or viewer.is_running()) and cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        time.sleep(0.01)
                        continue

                    if args.mirror:
                        frame = cv2.flip(frame, 1)

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    timestamp_ms = int(time.time() * 1000)
                    landmarks, raw_results = pose.detect(rgb, timestamp_ms)

                    status = "Pose: tracking" if landmarks else "Pose: not found"

                    target = None
                    diagnostics = None
                    if landmarks:
                        mapped = _map_landmarks_to_joints(landmarks, cfg, frame.shape)
                        if mapped is not None:
                            target, diagnostics = mapped

                    if target is None:
                        if cfg.lost_action == "home":
                            target = home_joints
                        else:
                            target = last_joints if last_joints is not None else home_joints
                            status = "Pose: holding"

                    smoothed = _smooth_joints(last_joints, target, cfg)
                    last_joints = smoothed

                    if not args.preview_only:
                        arm.move_to_joint_positions(smoothed.tolist())
                        # Step simulation multiple times for more responsive control
                        for _ in range(10):
                            arm.step()
                        if viewer is not None:
                            viewer.sync()

                    if gui_enabled:
                        frame.flags.writeable = True
                        if HAS_SOLUTIONS and raw_results is not None and raw_results.pose_landmarks:
                            DRAW_UTILS.draw_landmarks(
                                frame,
                                raw_results.pose_landmarks,
                                POSE_CONNECTIONS,
                                landmark_drawing_spec=DRAW_STYLES.get_default_pose_landmarks_style(),
                            )
                        elif landmarks is not None:
                            _draw_arm_landmarks(frame, landmarks, cfg.side)
                        _draw_status(frame, f"{status} | {args.side} arm | ESC to quit")
                        if diagnostics is not None:
                            torso_note = "" if diagnostics.torso_ok else " (torso fallback)"
                            _draw_status(
                                frame,
                                "torso_yaw={:.2f} pitch={:.2f} elbow={:.2f}{}".format(
                                    diagnostics.torso_yaw,
                                    diagnostics.shoulder_pitch,
                                    diagnostics.elbow_flex,
                                    torso_note,
                                ),
                                y=48,
                                color=(255, 255, 255),
                            )
                            _draw_status(
                                frame,
                                "targets={}".format(np.round(smoothed[:3], 2)),
                                y=72,
                                color=(255, 200, 100),
                            )
                        try:
                            cv2.imshow("Piper Vision Control", frame)
                            if cv2.waitKey(1) & 0xFF == 27:
                                break
                        except cv2.error as exc:
                            print(f"OpenCV preview failed ({exc}). Disabling camera window.")
                            gui_enabled = False
                            cv2.destroyAllWindows()
                    else:
                        # Prevent a tight loop when GUI is disabled.
                        time.sleep(0.005)

                    if args.debug and frame_idx % max(args.print_every, 1) == 0:
                        if landmarks is None:
                            print(f"[frame {frame_idx}] {status}")
                        else:
                            w = landmarks[RIGHT_WRIST_IDX if cfg.side == "right" else LEFT_WRIST_IDX]
                            s_idx = RIGHT_SHOULDER_IDX if cfg.side == "right" else LEFT_SHOULDER_IDX
                            e_idx = RIGHT_ELBOW_IDX if cfg.side == "right" else LEFT_ELBOW_IDX
                            w_idx = RIGHT_WRIST_IDX if cfg.side == "right" else LEFT_WRIST_IDX
                            vis = (
                                _landmark_visibility(landmarks, s_idx),
                                _landmark_visibility(landmarks, e_idx),
                                _landmark_visibility(landmarks, w_idx),
                            )
                            diag_text = ""
                            if diagnostics is not None:
                                diag_text = (
                                    f" yaw={diagnostics.shoulder_yaw:.2f}"
                                    f" pitch={diagnostics.shoulder_pitch:.2f}"
                                    f" elbow={diagnostics.elbow_flex:.2f}"
                                )
                            print(
                                f"[frame {frame_idx}] {status} "
                                f"wrist=({w.x:.2f},{w.y:.2f},{getattr(w, 'z', 0.0):.2f}) "
                                f"vis=({vis[0]:.2f},{vis[1]:.2f},{vis[2]:.2f}) "
                                f"joints={smoothed.round(2)}{diag_text}"
                            )

                    if log_writer and frame_idx % max(args.log_every, 1) == 0:
                        if landmarks is None:
                            wx = wy = wz = float("nan")
                            shoulder_vis = elbow_vis = wrist_vis = float("nan")
                            shoulder_yaw = shoulder_pitch = elbow_flex = float("nan")
                            torso_ok = 0
                        else:
                            w_idx = RIGHT_WRIST_IDX if cfg.side == "right" else LEFT_WRIST_IDX
                            s_idx = RIGHT_SHOULDER_IDX if cfg.side == "right" else LEFT_SHOULDER_IDX
                            e_idx = RIGHT_ELBOW_IDX if cfg.side == "right" else LEFT_ELBOW_IDX
                            w = landmarks[w_idx]
                            wx = float(w.x)
                            wy = float(w.y)
                            wz = float(getattr(w, "z", 0.0))
                            shoulder_vis = _landmark_visibility(landmarks, s_idx)
                            elbow_vis = _landmark_visibility(landmarks, e_idx)
                            wrist_vis = _landmark_visibility(landmarks, w_idx)
                            if diagnostics is None:
                                shoulder_yaw = shoulder_pitch = elbow_flex = torso_yaw = float("nan")
                                torso_ok = 0
                            else:
                                shoulder_yaw = diagnostics.shoulder_yaw
                                shoulder_pitch = diagnostics.shoulder_pitch
                                elbow_flex = diagnostics.elbow_flex
                                torso_yaw = diagnostics.torso_yaw
                                torso_ok = int(diagnostics.torso_ok)

                        actual = arm.get_joint_positions() if not args.preview_only else np.full(6, np.nan)
                        log_writer.writerow(
                            [
                                frame_idx,
                                time.time(),
                                status,
                                wx,
                                wy,
                                wz,
                                shoulder_vis,
                                elbow_vis,
                                wrist_vis,
                                shoulder_yaw,
                                shoulder_pitch,
                                elbow_flex,
                                torso_yaw,
                                torso_ok,
                                *smoothed.tolist(),
                                *actual.tolist(),
                            ]
                        )
                    frame_idx += 1
        finally:
            cap.release()
            if gui_enabled:
                cv2.destroyAllWindows()
            if log_file:
                log_file.flush()
                log_file.close()


if __name__ == "__main__":
    main()
