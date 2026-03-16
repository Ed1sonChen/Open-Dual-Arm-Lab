#!/usr/bin/env python3
import asyncio
import websockets
import json
import time
import sys
import threading
import math
import csv
import os
import select
import termios
import tty
import atexit
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI


# ==========================================
# 1. Configuration
# ==========================================

IP_LEFT = "your robot ip here"
SCALE = 500.0

# Replace with your real camera serial numbers
FRONT_CAM_SERIAL = ""
WRIST_CAM_SERIAL = ""

DATASET_ROOT = Path("./dataset/raw")

LIMITS = {
    "X_MIN": 100, "X_MAX": 700,
    "Y_MIN": -600, "Y_MAX": 600,
    "Z_MIN": 0,   "Z_MAX": 700
}

MAX_STEP_TRANSLATION_MM = 8.0
MAX_STEP_ROTATION_DEG = 5.0
PACKET_TIMEOUT_SEC = 0.20

ROBOT_CONTROL_HZ = 100.0
ROBOT_STATE_HZ = 30.0

CAMERA_FPS = 30
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
ENABLE_DEPTH = False   # no depth display / no depth saving

# Manual camera settings (only effective if supported by your RealSense model)
COLOR_EXPOSURE = 166
COLOR_GAIN = 64
COLOR_WHITE_BALANCE = 3000

GRIPPER_OPEN = 850
GRIPPER_CLOSE = 0

# Reset / home behavior
SAFE_LIFT_Z = 220.0         # mm, safe lift height before going home
RESET_SPEED = 80.0          # xArm move speed
RESET_ACC = 800.0           # xArm move acceleration
AUTO_RETURN_HOME_AFTER_STOP = False   # set True if you want s/f to auto home

# IMPORTANT: replace this with your own safe home joint pose (degrees)
HOME_JOINT = []

# Display
SHOW_CAMERA_WINDOWS = True
DISPLAY_SCALE = 1.0
DISPLAY_WINDOW_NAME = "Front RGB | Wrist RGB"


# ==========================================
# 2. One Euro Filter 3D
# ==========================================
class OneEuroFilter3D:
    def __init__(self, mincutoff=1.0, beta=0.005, dcutoff=1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, is_angle=False):
        t_now = time.time()
        if self.t_prev is None:
            self.t_prev = t_now
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x.copy()

        dt = t_now - self.t_prev
        if dt <= 0:
            dt = 1e-4

        dx = x - self.x_prev
        if is_angle:
            dx = (dx + 180) % 360 - 180

        dx = dx / dt

        a_d = self.alpha(self.dcutoff, dt)
        edx = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.mincutoff + self.beta * np.abs(edx)
        a = self.alpha(cutoff, dt)

        if is_angle:
            diff = x - self.x_prev
            diff = (diff + 180) % 360 - 180
            x_hat = self.x_prev + a * diff
            x_hat = (x_hat + 180) % 360 - 180
        else:
            x_hat = self.x_prev + a * (x - self.x_prev)

        self.x_prev = x_hat.copy()
        self.dx_prev = edx.copy()
        self.t_prev = t_now
        return x_hat


# ==========================================
# 3. Helper functions
# ==========================================
def now_ns():
    return time.time_ns()


def clamp_vector_norm(vec, max_norm):
    norm = np.linalg.norm(vec)
    if norm <= max_norm or norm < 1e-9:
        return vec
    return vec * (max_norm / norm)


def clamp_rotation_delta(delta_rot: R, max_deg: float) -> R:
    rotvec = delta_rot.as_rotvec()
    angle = np.linalg.norm(rotvec)
    max_rad = np.deg2rad(max_deg)

    if angle <= max_rad or angle < 1e-9:
        return delta_rot

    rotvec_clamped = rotvec * (max_rad / angle)
    return R.from_rotvec(rotvec_clamped)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def next_episode_id(root: Path) -> str:
    ensure_dir(root)
    existing = sorted([p.name for p in root.glob("episode_*") if p.is_dir()])
    if not existing:
        return "episode_000001"
    last_num = int(existing[-1].split("_")[-1])
    return f"episode_{last_num + 1:06d}"


def save_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def rs_intrinsics_to_dict(intrin):
    return {
        "width": intrin.width,
        "height": intrin.height,
        "ppx": intrin.ppx,
        "ppy": intrin.ppy,
        "fx": intrin.fx,
        "fy": intrin.fy,
        "model": str(intrin.model),
        "coeffs": list(intrin.coeffs),
    }


def rs_extrinsics_to_dict(extrin):
    return {
        "rotation_row_major": list(extrin.rotation),
        "translation_m": list(extrin.translation),
    }


def overlay_frame_info(img, camera_name, frame_id, timestamp_ns):
    vis = img.copy()
    ts_sec = timestamp_ns / 1e9
    text1 = f"{camera_name} | frame_id: {frame_id}"
    text2 = f"timestamp_ns: {timestamp_ns}"
    text3 = f"timestamp_sec: {ts_sec:.6f}"

    y0 = 28
    dy = 28
    for i, txt in enumerate([text1, text2, text3]):
        y = y0 + i * dy
        cv2.putText(
            vis, txt, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            vis, txt, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0), 2, cv2.LINE_AA
        )
    return vis


def stack_rgb_views(front_img, wrist_img):
    if front_img is None and wrist_img is None:
        return None

    black = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

    if front_img is None:
        front_img = black.copy()
        cv2.putText(front_img, "front: no frame", (20, CAMERA_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    if wrist_img is None:
        wrist_img = black.copy()
        cv2.putText(wrist_img, "wrist: no frame", (20, CAMERA_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    if front_img.shape[:2] != (CAMERA_HEIGHT, CAMERA_WIDTH):
        front_img = cv2.resize(front_img, (CAMERA_WIDTH, CAMERA_HEIGHT))
    if wrist_img.shape[:2] != (CAMERA_HEIGHT, CAMERA_WIDTH):
        wrist_img = cv2.resize(wrist_img, (CAMERA_WIDTH, CAMERA_HEIGHT))

    combined = np.hstack([front_img, wrist_img])
    return combined


# ==========================================
# 4. Recorder
# ==========================================
class EpisodeRecorder:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        ensure_dir(self.root_dir)

        self.active = False
        self.lock = threading.Lock()

        self.episode_id = None
        self.episode_dir = None

        self.front_rgb_dir = None
        self.wrist_rgb_dir = None

        self.robot_state_file = None
        self.robot_state_writer = None

        self.command_file = None
        self.command_writer = None

        self.meta = {}
        self.frame_counts = {
            "front_rgb": 0,
            "wrist_rgb": 0,
        }
        self.last_saved_frame_id = {
            "front": -1,
            "wrist": -1,
        }

    def start(self, object_id="object_v1", task_name="flip_object"):
        with self.lock:
            if self.active:
                print("[WARN] Episode already active.")
                return False

            self.episode_id = next_episode_id(self.root_dir)
            self.episode_dir = self.root_dir / self.episode_id
            ensure_dir(self.episode_dir)

            self.front_rgb_dir = self.episode_dir / "front_rgb"
            self.wrist_rgb_dir = self.episode_dir / "wrist_rgb"

            for d in [self.front_rgb_dir, self.wrist_rgb_dir]:
                ensure_dir(d)

            self.robot_state_file = open(self.episode_dir / "robot_state.csv", "w", newline="", encoding="utf-8")
            self.robot_state_writer = csv.DictWriter(
                self.robot_state_file,
                fieldnames=[
                    "timestamp_ns",
                    "j1", "j2", "j3", "j4", "j5", "j6",
                    "dj1", "dj2", "dj3", "dj4", "dj5", "dj6",
                    "tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz",
                    "gripper_pos"
                ]
            )
            self.robot_state_writer.writeheader()

            self.command_file = open(self.episode_dir / "teleop_command.csv", "w", newline="", encoding="utf-8")
            self.command_writer = csv.DictWriter(
                self.command_file,
                fieldnames=[
                    "timestamp_ns",
                    "cmd_tcp_x", "cmd_tcp_y", "cmd_tcp_z",
                    "cmd_tcp_rx", "cmd_tcp_ry", "cmd_tcp_rz",
                    "cmd_gripper",
                    "vr_px", "vr_py", "vr_pz",
                    "vr_qx", "vr_qy", "vr_qz", "vr_qw",
                    "trigger", "grip"
                ]
            )
            self.command_writer.writeheader()

            self.meta = {
                "episode_id": self.episode_id,
                "task_name": task_name,
                "object_id": object_id,
                "success": None,
                "failure_reason": None,
                "operator": "quest3",
                "start_time_ns": now_ns(),
                "start_from_home": True,
                "home_joint_deg": HOME_JOINT,
            }

            self.frame_counts = {
                "front_rgb": 0,
                "wrist_rgb": 0,
            }
            self.last_saved_frame_id = {
                "front": -1,
                "wrist": -1,
            }

            calib_src_dir = self.root_dir / "camera_calibration"
            calib_dst_dir = self.episode_dir / "camera_calibration"
            ensure_dir(calib_dst_dir)
            if calib_src_dir.exists():
                for p in calib_src_dir.glob("*.json"):
                    try:
                        with open(p, "r", encoding="utf-8") as fsrc:
                            data = json.load(fsrc)
                        with open(calib_dst_dir / p.name, "w", encoding="utf-8") as fdst:
                            json.dump(data, fdst, indent=2)
                    except Exception as e:
                        print(f"[WARN] Failed to copy calibration file {p}: {e}")

            self.active = True
            print(f"[INFO] Start recording: {self.episode_id}")
            return True

    def stop(self, success: bool, failure_reason=None):
        with self.lock:
            if not self.active:
                print("[WARN] No active episode.")
                return False

            self.meta["success"] = success
            self.meta["failure_reason"] = failure_reason
            self.meta["end_time_ns"] = now_ns()

            with open(self.episode_dir / "episode_meta.json", "w", encoding="utf-8") as f:
                json.dump(self.meta, f, indent=2)

            self.robot_state_file.flush()
            self.command_file.flush()
            self.robot_state_file.close()
            self.command_file.close()

            print(f"[INFO] Stop recording: {self.episode_id}, success={success}, failure_reason={failure_reason}")

            self.active = False
            self.episode_id = None
            self.episode_dir = None
            return True

    def write_robot_state(self, row):
        with self.lock:
            if self.active:
                self.robot_state_writer.writerow(row)

    def write_command(self, row):
        with self.lock:
            if self.active:
                self.command_writer.writerow(row)

    def save_camera_frame(self, camera_name, frame_id, timestamp_ns, color_bgr):
        with self.lock:
            if not self.active:
                return

            if self.last_saved_frame_id[camera_name] == frame_id:
                return
            self.last_saved_frame_id[camera_name] = frame_id

            if camera_name == "front":
                rgb_dir = self.front_rgb_dir
                rgb_idx = self.frame_counts["front_rgb"]
                self.frame_counts["front_rgb"] += 1
            elif camera_name == "wrist":
                rgb_dir = self.wrist_rgb_dir
                rgb_idx = self.frame_counts["wrist_rgb"]
                self.frame_counts["wrist_rgb"] += 1
            else:
                return

            rgb_path = rgb_dir / f"{rgb_idx:06d}_{timestamp_ns}.png"
            cv2.imwrite(str(rgb_path), color_bgr)


# ==========================================
# 5. RealSense camera thread
# ==========================================
class RealSenseReader:
    def __init__(
        self,
        name,
        serial,
        width,
        height,
        fps,
        enable_depth=False,
        dataset_root=Path("./dataset/raw"),
        manual_exposure=166,
        manual_gain=64,
        manual_white_balance=3000,
    ):
        self.name = name
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        self.dataset_root = Path(dataset_root)
        self.manual_exposure = manual_exposure
        self.manual_gain = manual_gain
        self.manual_white_balance = manual_white_balance

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.serial)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.enable_depth:
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self.align = rs.align(rs.stream.color)
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        self.latest = None
        self.frame_id = 0
        self.profile = None

    def start(self):
        self.profile = self.pipeline.start(self.config)
        self._configure_sensor_options()
        self._save_camera_calibration()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"[INFO] RealSense {self.name} ({self.serial}) started.")

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        try:
            self.pipeline.stop()
        except Exception:
            pass
        print(f"[INFO] RealSense {self.name} stopped.")

    def _configure_sensor_options(self):
        try:
            device = self.profile.get_device()
            sensors = device.query_sensors()

            color_sensor = None
            for s in sensors:
                sname = s.get_info(rs.camera_info.name)
                if "RGB Camera" in sname:
                    color_sensor = s
                    break

            if color_sensor is not None:
                if color_sensor.supports(rs.option.enable_auto_exposure):
                    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                if color_sensor.supports(rs.option.enable_auto_white_balance):
                    color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
                if color_sensor.supports(rs.option.exposure):
                    color_sensor.set_option(rs.option.exposure, float(self.manual_exposure))
                if color_sensor.supports(rs.option.gain):
                    color_sensor.set_option(rs.option.gain, float(self.manual_gain))
                if color_sensor.supports(rs.option.white_balance):
                    color_sensor.set_option(rs.option.white_balance, float(self.manual_white_balance))

                print(
                    f"[INFO] {self.name}: auto exposure OFF, auto white balance OFF, "
                    f"exposure={self.manual_exposure}, gain={self.manual_gain}, wb={self.manual_white_balance}"
                )

        except Exception as e:
            print(f"[WARN] Failed to configure sensor options for {self.name}: {e}")

    def _save_camera_calibration(self):
        cam_dir = self.dataset_root / "camera_calibration"
        ensure_dir(cam_dir)

        try:
            color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
            color_intr = color_stream.get_intrinsics()

            calib = {
                "camera_name": self.name,
                "serial": self.serial,
                "color_intrinsics": rs_intrinsics_to_dict(color_intr),
                "resolution": {
                    "width": self.width,
                    "height": self.height,
                },
                "fps": self.fps,
                "auto_exposure": False,
                "auto_white_balance": False,
                "manual_exposure": self.manual_exposure,
                "manual_gain": self.manual_gain,
                "manual_white_balance": self.manual_white_balance,
                "note": (
                    "This file stores RGB intrinsics. "
                    "Extrinsics between front camera and wrist camera are NOT provided by factory calibration "
                    "and require external multi-camera calibration."
                )
            }

            if self.enable_depth:
                depth_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
                depth_intr = depth_stream.get_intrinsics()
                depth_to_color = depth_stream.get_extrinsics_to(color_stream)
                color_to_depth = color_stream.get_extrinsics_to(depth_stream)
                calib["depth_intrinsics"] = rs_intrinsics_to_dict(depth_intr)
                calib["extrinsics_depth_to_color"] = rs_extrinsics_to_dict(depth_to_color)
                calib["extrinsics_color_to_depth"] = rs_extrinsics_to_dict(color_to_depth)

            out_path = cam_dir / f"{self.name}_{self.serial}.json"
            save_json(out_path, calib)
            print(f"[INFO] Saved calibration for {self.name} to {out_path}")

        except Exception as e:
            print(f"[WARN] Failed to save calibration for {self.name}: {e}")

    def _loop(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames()
                if self.enable_depth:
                    frames = self.align.process(frames)

                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                color = np.asanyarray(color_frame.get_data())

                with self.lock:
                    self.latest = {
                        "frame_id": self.frame_id,
                        "timestamp_ns": now_ns(),
                        "color_bgr": color
                    }
                    self.frame_id += 1

            except Exception as e:
                print(f"[WARN] RealSense reader error ({self.name}, {self.serial}): {e}")
                time.sleep(0.01)

    def get_latest(self):
        with self.lock:
            if self.latest is None:
                return None
            return {
                "frame_id": self.latest["frame_id"],
                "timestamp_ns": self.latest["timestamp_ns"],
                "color_bgr": self.latest["color_bgr"].copy(),
            }


# ==========================================
# 6. Shared teleop / robot states
# ==========================================

M_mat = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0]
])
M_transform = R.from_matrix(M_mat)
M_transform_inv = M_transform.inv()

print("Connecting to LEFT xArm...")
try:
    arm_left = XArmAPI(IP_LEFT)
except Exception as e:
    print(f"Robot connection failed: {e}")
    sys.exit()

arm_left.clean_error()
arm_left.motion_enable(enable=True)
arm_left.set_mode(1)
arm_left.set_state(0)
time.sleep(0.5)

try:
    arm_left.set_gripper_enable(True)
    arm_left.set_gripper_mode(0)
except Exception as e:
    print(f"[WARN] Gripper init warning: {e}")

print("✅ Left arm initialization successful. Right controller -> Left arm teleoperation active.")

prev_vr_pos = None
prev_vr_quat = None
last_gripper_pos = -1

raw_target_xyz = None
raw_target_rpy = None
last_packet_time = 0.0
timeout_triggered = False
last_vr_packet = None

servo_enabled = True

data_lock = threading.Lock()
home_lock = threading.Lock()

pos_filter = OneEuroFilter3D(mincutoff=0.8, beta=0.04)
rot_filter = OneEuroFilter3D(mincutoff=0.5, beta=0.02)

recorder = EpisodeRecorder(DATASET_ROOT)

front_cam = None
wrist_cam = None


# ==========================================
# 7. Reset utilities
# ==========================================
def reset_teleop_state():
    global prev_vr_pos, prev_vr_quat
    global raw_target_xyz, raw_target_rpy
    global last_packet_time, timeout_triggered
    global last_vr_packet

    with data_lock:
        prev_vr_pos = None
        prev_vr_quat = None
        raw_target_xyz = None
        raw_target_rpy = None
        last_packet_time = 0.0
        timeout_triggered = False
        last_vr_packet = None

    pos_filter.reset()
    rot_filter.reset()


def move_robot_to_home(arm):
    global servo_enabled

    with home_lock:
        print("[INFO] Returning robot to home pose...")

        servo_enabled = False
        reset_teleop_state()
        time.sleep(0.05)

        try:
            arm.set_mode(0)
            arm.set_state(0)
            time.sleep(0.05)

            code, current_tcp = arm.get_position(is_radian=False)
            if code == 0:
                safe_z = max(float(current_tcp[2]), SAFE_LIFT_Z)

                arm.set_position(
                    x=float(current_tcp[0]),
                    y=float(current_tcp[1]),
                    z=float(safe_z),
                    roll=float(current_tcp[3]),
                    pitch=float(current_tcp[4]),
                    yaw=float(current_tcp[5]),
                    speed=RESET_SPEED,
                    mvacc=RESET_ACC,
                    wait=True
                )

            arm.set_servo_angle(
                angle=HOME_JOINT,
                speed=RESET_SPEED,
                mvacc=RESET_ACC,
                wait=True,
                is_radian=False
            )

            print("[INFO] Robot reached home pose.")

        except Exception as e:
            print(f"[WARN] move_robot_to_home failed: {e}")

        finally:
            try:
                arm.set_mode(1)
                arm.set_state(0)
                time.sleep(0.05)
            except Exception as e:
                print(f"[WARN] Failed to restore servo mode: {e}")

            servo_enabled = True
            print("[INFO] Servo teleop re-enabled.")


# ==========================================
# 8. Robot state recorder thread
# ==========================================
def robot_state_record_loop(arm):
    dt = 1.0 / ROBOT_STATE_HZ
    while True:
        try:
            if recorder.active:
                code_j, joints = arm.get_servo_angle(is_radian=False)
                code_p, tcp = arm.get_position(is_radian=False)

                if code_j == 0 and code_p == 0:
                    try:
                        joint_vel = arm.realtime_joint_speeds
                        if joint_vel is None:
                            joint_vel = [0.0] * 6
                    except Exception:
                        joint_vel = [0.0] * 6

                    try:
                        code_g, grip_pos = arm.get_gripper_position()
                        if code_g != 0:
                            grip_pos = 0.0
                    except Exception:
                        grip_pos = 0.0

                    recorder.write_robot_state({
                        "timestamp_ns": now_ns(),
                        "j1": joints[0], "j2": joints[1], "j3": joints[2],
                        "j4": joints[3], "j5": joints[4], "j6": joints[5],
                        "dj1": joint_vel[0], "dj2": joint_vel[1], "dj3": joint_vel[2],
                        "dj4": joint_vel[3], "dj5": joint_vel[4], "dj6": joint_vel[5],
                        "tcp_x": tcp[0], "tcp_y": tcp[1], "tcp_z": tcp[2],
                        "tcp_rx": tcp[3], "tcp_ry": tcp[4], "tcp_rz": tcp[5],
                        "gripper_pos": grip_pos
                    })
        except Exception as e:
            print(f"[WARN] robot_state_record_loop error: {e}")

        time.sleep(dt)


# ==========================================
# 9. Camera recorder thread
# ==========================================
def camera_record_loop():
    while True:
        try:
            if recorder.active:
                front = front_cam.get_latest() if front_cam is not None else None
                wrist = wrist_cam.get_latest() if wrist_cam is not None else None

                if front is not None:
                    recorder.save_camera_frame(
                        "front",
                        frame_id=front["frame_id"],
                        timestamp_ns=front["timestamp_ns"],
                        color_bgr=front["color_bgr"]
                    )

                if wrist is not None:
                    recorder.save_camera_frame(
                        "wrist",
                        frame_id=wrist["frame_id"],
                        timestamp_ns=wrist["timestamp_ns"],
                        color_bgr=wrist["color_bgr"]
                    )
        except Exception as e:
            print(f"[WARN] camera_record_loop error: {e}")

        time.sleep(0.005)


# ==========================================
# 9.5 Camera display thread
# ==========================================
def camera_display_loop():
    if not SHOW_CAMERA_WINDOWS:
        return

    cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        try:
            front = front_cam.get_latest() if front_cam is not None else None
            wrist = wrist_cam.get_latest() if wrist_cam is not None else None

            front_vis = None
            wrist_vis = None

            if front is not None:
                front_vis = overlay_frame_info(
                    front["color_bgr"], "front_rgb", front["frame_id"], front["timestamp_ns"]
                )

            if wrist is not None:
                wrist_vis = overlay_frame_info(
                    wrist["color_bgr"], "wrist_rgb", wrist["frame_id"], wrist["timestamp_ns"]
                )

            combined = stack_rgb_views(front_vis, wrist_vis)
            if combined is not None:
                if DISPLAY_SCALE != 1.0:
                    combined = cv2.resize(combined, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
                cv2.imshow(DISPLAY_WINDOW_NAME, combined)

            cv2.waitKey(1)

        except Exception as e:
            print(f"[WARN] camera_display_loop error: {e}")
            time.sleep(0.01)


# ==========================================
# 10. High-frequency servo loop
# ==========================================
def robot_control_loop(arm):
    global raw_target_xyz, raw_target_rpy
    global last_packet_time, timeout_triggered
    global servo_enabled

    dt = 1.0 / ROBOT_CONTROL_HZ

    while True:
        if not servo_enabled:
            time.sleep(dt)
            continue

        now = time.time()

        with data_lock:
            t_xyz = None if raw_target_xyz is None else raw_target_xyz.copy()
            t_rpy = None if raw_target_rpy is None else raw_target_rpy.copy()
            pkt_time = last_packet_time
            vr_pkt = None if last_vr_packet is None else last_vr_packet.copy()

        if pkt_time > 0 and (now - pkt_time) > PACKET_TIMEOUT_SEC:
            if not timeout_triggered:
                print(f"⚠️ Right controller timeout: no packet for {now - pkt_time:.3f}s, servo paused.")
                timeout_triggered = True

            reset_teleop_state()
            time.sleep(dt)
            continue

        if pkt_time > 0 and timeout_triggered:
            timeout_triggered = False
            print("✅ Right controller packet stream recovered.")

        if t_xyz is not None and t_rpy is not None:
            f_xyz = pos_filter(np.array(t_xyz), is_angle=False)
            f_rpy = rot_filter(np.array(t_rpy), is_angle=True)

            final_target = [
                float(f_xyz[0]), float(f_xyz[1]), float(f_xyz[2]),
                float(f_rpy[0]), float(f_rpy[1]), float(f_rpy[2])
            ]

            try:
                arm.set_servo_cartesian(final_target)
            except Exception as e:
                print(f"[WARN] set_servo_cartesian failed: {e}")

            if recorder.active and vr_pkt is not None:
                recorder.write_command({
                    "timestamp_ns": now_ns(),
                    "cmd_tcp_x": final_target[0],
                    "cmd_tcp_y": final_target[1],
                    "cmd_tcp_z": final_target[2],
                    "cmd_tcp_rx": final_target[3],
                    "cmd_tcp_ry": final_target[4],
                    "cmd_tcp_rz": final_target[5],
                    "cmd_gripper": vr_pkt["cmd_gripper"],
                    "vr_px": vr_pkt["vr_px"],
                    "vr_py": vr_pkt["vr_py"],
                    "vr_pz": vr_pkt["vr_pz"],
                    "vr_qx": vr_pkt["vr_qx"],
                    "vr_qy": vr_pkt["vr_qy"],
                    "vr_qz": vr_pkt["vr_qz"],
                    "vr_qw": vr_pkt["vr_qw"],
                    "trigger": vr_pkt["trigger"],
                    "grip": vr_pkt["grip"],
                })

        time.sleep(dt)


# ==========================================
# 11. Process RIGHT controller packets
# ==========================================
def process_right_controller_for_left_arm(vr_data, arm):
    global prev_vr_pos, prev_vr_quat
    global raw_target_xyz, raw_target_rpy
    global last_gripper_pos, last_packet_time
    global last_vr_packet
    global servo_enabled

    if not servo_enabled:
        return

    vr_pos = vr_data["position"]
    vr_q = vr_data["quaternion"]
    trigger = vr_data["trigger"]
    grip = vr_data["grip"]

    with data_lock:
        last_packet_time = time.time()

    gripper_target = int(GRIPPER_OPEN - (trigger * (GRIPPER_OPEN - GRIPPER_CLOSE)))
    if abs(gripper_target - last_gripper_pos) > 15:
        try:
            arm.set_gripper_position(gripper_target, wait=False)
        except Exception:
            pass
        last_gripper_pos = gripper_target

    with data_lock:
        last_vr_packet = {
            "vr_px": vr_pos["x"],
            "vr_py": vr_pos["y"],
            "vr_pz": vr_pos["z"],
            "vr_qx": vr_q["x"],
            "vr_qy": vr_q["y"],
            "vr_qz": vr_q["z"],
            "vr_qw": vr_q["w"],
            "trigger": trigger,
            "grip": grip,
            "cmd_gripper": gripper_target
        }

        if grip > 0.5:
            curr_vr_quat = R.from_quat([
                vr_q["x"], vr_q["y"], vr_q["z"], vr_q["w"]
            ])

            if prev_vr_pos is None:
                code, current_tcp = arm.get_position()
                if code == 0:
                    raw_target_xyz = [
                        current_tcp[0], current_tcp[1], current_tcp[2]
                    ]
                    raw_target_rpy = [
                        current_tcp[3], current_tcp[4], current_tcp[5]
                    ]
                prev_vr_pos = vr_pos.copy()
                prev_vr_quat = curr_vr_quat
                return

            dx_vr = vr_pos["x"] - prev_vr_pos["x"]
            dy_vr = vr_pos["y"] - prev_vr_pos["y"]
            dz_vr = vr_pos["z"] - prev_vr_pos["z"]

            delta_xyz_robot = np.array([
                -dz_vr * SCALE,
                -dx_vr * SCALE,
                +dy_vr * SCALE
            ], dtype=float)

            delta_xyz_robot = clamp_vector_norm(delta_xyz_robot, MAX_STEP_TRANSLATION_MM)

            new_xyz = np.array(raw_target_xyz, dtype=float) + delta_xyz_robot
            new_xyz[0] = max(LIMITS["X_MIN"], min(new_xyz[0], LIMITS["X_MAX"]))
            new_xyz[1] = max(LIMITS["Y_MIN"], min(new_xyz[1], LIMITS["Y_MAX"]))
            new_xyz[2] = max(LIMITS["Z_MIN"], min(new_xyz[2], LIMITS["Z_MAX"]))
            raw_target_xyz = new_xyz.tolist()

            delta_R_vr = curr_vr_quat * prev_vr_quat.inv()
            delta_R_rob = M_transform * delta_R_vr * M_transform_inv
            delta_R_rob = clamp_rotation_delta(delta_R_rob, MAX_STEP_ROTATION_DEG)

            current_R_rob = R.from_euler("xyz", raw_target_rpy, degrees=True)
            new_R_rob = delta_R_rob * current_R_rob
            raw_target_rpy = new_R_rob.as_euler("xyz", degrees=True).tolist()

            prev_vr_pos = vr_pos.copy()
            prev_vr_quat = curr_vr_quat

        else:
            prev_vr_pos = None
            prev_vr_quat = None
            raw_target_xyz = None
            raw_target_rpy = None

            pos_filter.reset()
            rot_filter.reset()


# ==========================================
# 12. Non-blocking keyboard utilities
# ==========================================
_keyboard_fd = None
_keyboard_old_settings = None


def setup_keyboard_nonblocking():
    global _keyboard_fd, _keyboard_old_settings
    if not sys.stdin.isatty():
        print("[WARN] stdin is not a TTY. Single-key keyboard control may not work.")
        return

    _keyboard_fd = sys.stdin.fileno()
    _keyboard_old_settings = termios.tcgetattr(_keyboard_fd)
    tty.setcbreak(_keyboard_fd)


def restore_keyboard():
    global _keyboard_fd, _keyboard_old_settings
    try:
        if _keyboard_fd is not None and _keyboard_old_settings is not None:
            termios.tcsetattr(_keyboard_fd, termios.TCSADRAIN, _keyboard_old_settings)
    except Exception:
        pass


def get_key_nonblocking():
    if _keyboard_fd is None:
        return None
    dr, _, _ = select.select([sys.stdin], [], [], 0.05)
    if dr:
        ch = sys.stdin.read(1)
        return ch.lower()
    return None


# ==========================================
# 13. Keyboard control thread
# ==========================================
def keyboard_loop():
    print("\n[Keyboard Commands]")
    print("  h  -> return robot to home/reset pose")
    print("  r  -> start recording")
    print("  s  -> stop recording as success")
    print("  f  -> stop recording as failure")
    print("  q  -> quit")
    print("  (press single key directly, no Enter needed)\n")

    while True:
        try:
            cmd = get_key_nonblocking()
            if cmd is None:
                continue

            if cmd == "h":
                print("[KEY] h")
                move_robot_to_home(arm_left)

            elif cmd == "r":
                print("[KEY] r")
                recorder.start(object_id="flip_object_v1", task_name="flip_object")

            elif cmd == "s":
                print("[KEY] s")
                stopped = recorder.stop(success=True, failure_reason=None)
                if stopped and AUTO_RETURN_HOME_AFTER_STOP:
                    move_robot_to_home(arm_left)

            elif cmd == "f":
                print("[KEY] f")
                stopped = recorder.stop(success=False, failure_reason="manual_failure")
                if stopped and AUTO_RETURN_HOME_AFTER_STOP:
                    move_robot_to_home(arm_left)

            elif cmd == "q":
                print("[KEY] q")
                print("[INFO] Quit requested.")
                restore_keyboard()
                os._exit(0)

        except Exception as e:
            print(f"[WARN] keyboard_loop error: {e}")
            time.sleep(0.05)


# ==========================================
# 14. WebSocket server
# ==========================================
async def handle_vr_data(websocket):
    print("🚀 Quest 3 teleoperation session start! (RIGHT controller -> LEFT arm)")
    try:
        async for message in websocket:
            data = json.loads(message)
            if "right" in data:
                process_right_controller_for_left_arm(data["right"], arm_left)

    except websockets.exceptions.ConnectionClosed:
        print("❌ Quest 3 connection closed")
        reset_teleop_state()
        try:
            arm_left.set_state(4)
        except Exception:
            pass

    except Exception as e:
        print(f"❌ WebSocket handler error: {e}")
        reset_teleop_state()
        try:
            arm_left.set_state(4)
        except Exception:
            pass


# ==========================================
# 15. Main
# ==========================================
async def main():
    global front_cam, wrist_cam

    ensure_dir(DATASET_ROOT)
    setup_keyboard_nonblocking()
    atexit.register(restore_keyboard)

    print("[INFO] Starting RealSense cameras...")
    front_cam = RealSenseReader(
        name="front",
        serial=FRONT_CAM_SERIAL,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
        enable_depth=ENABLE_DEPTH,
        dataset_root=DATASET_ROOT,
        manual_exposure=COLOR_EXPOSURE,
        manual_gain=COLOR_GAIN,
        manual_white_balance=COLOR_WHITE_BALANCE,
    )
    wrist_cam = RealSenseReader(
        name="wrist",
        serial=WRIST_CAM_SERIAL,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
        enable_depth=ENABLE_DEPTH,
        dataset_root=DATASET_ROOT,
        manual_exposure=COLOR_EXPOSURE,
        manual_gain=COLOR_GAIN,
        manual_white_balance=COLOR_WHITE_BALANCE,
    )

    front_cam.start()
    wrist_cam.start()

    threading.Thread(target=robot_control_loop, args=(arm_left,), daemon=True).start()
    threading.Thread(target=robot_state_record_loop, args=(arm_left,), daemon=True).start()
    threading.Thread(target=camera_record_loop, daemon=True).start()
    threading.Thread(target=camera_display_loop, daemon=True).start()
    threading.Thread(target=keyboard_loop, daemon=True).start()

    async with websockets.serve(handle_vr_data, "0.0.0.0", 8765):
        print("Waiting for VR client connection...")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        try:
            restore_keyboard()
        except Exception:
            pass

        try:
            if front_cam is not None:
                front_cam.stop()
            if wrist_cam is not None:
                wrist_cam.stop()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass