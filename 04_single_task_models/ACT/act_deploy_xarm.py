import os
import csv
import cv2
import time
import json
import math
import pickle
import shutil
import threading
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
import torch

from xarm.wrapper import XArmAPI
from policy import ACTPolicy


# =========================================================
# 1. Config
# =========================================================
IP_LEFT = "192.168.1.204"

FRONT_CAM_SERIAL = "336222076423"
WRIST_CAM_SERIAL = "346522070903"

FRONT_CAM_CONFIG_PATH = "dataset/raw/camera_calibration/front_336222076423.json"
WRIST_CAM_CONFIG_PATH = "dataset/raw/camera_calibration/wrist_346522070903.json"

CKPT_DIR = "./ckpts/xarm_flip_block_act"
CKPT_NAME = "policy_best.ckpt"

ROLLOUT_ROOT = Path("./rollout_logs_act_official")

CAMERA_NAMES = ["front", "wrist"]
STATE_DIM = 7
ACTION_DIM = 7

IMAGE_SIZE = (480, 640)   # (H, W)
CONTROL_HZ = 10.0         # 主控制循环（模型推理）频率
QUERY_FREQUENCY = 1       
TEMPORAL_AGG = True
CHUNK_SIZE = 20

ENABLE_GRIPPER = True
GRIPPER_MIN = 0.0
GRIPPER_MAX = 850.0

MAX_JOINT_STEP_DEG = np.array([3.0, 3.0, 3.0, 4.0, 4.0, 4.0], dtype=np.float32)

JOINT_LIMITS = {
    "J1_MIN": -360.0, "J1_MAX": 360.0,
    "J2_MIN": -360.0, "J2_MAX": 360.0,
    "J3_MIN": -360.0, "J3_MAX": 360.0,
    "J4_MIN": -360.0, "J4_MAX": 360.0,
    "J5_MIN": -360.0, "J5_MAX": 360.0,
    "J6_MIN": -360.0, "J6_MAX": 360.0,
}

SAFE_LIFT_Z = 220.0
RESET_SPEED = 80.0
RESET_ACC = 800.0
AUTO_RETURN_HOME_AFTER_STOP = True

HOME_JOINT = [-12.4, -17.6, -49.1, -3.1, 64.9, -13.4]


# =========================================================
# 2. Utils
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def now_ns():
    return time.time_ns()

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def clamp_joint(joint_deg):
    joint_deg = np.asarray(joint_deg, dtype=np.float32).copy()
    joint_deg[0] = clamp(joint_deg[0], JOINT_LIMITS["J1_MIN"], JOINT_LIMITS["J1_MAX"])
    joint_deg[1] = clamp(joint_deg[1], JOINT_LIMITS["J2_MIN"], JOINT_LIMITS["J2_MAX"])
    joint_deg[2] = clamp(joint_deg[2], JOINT_LIMITS["J3_MIN"], JOINT_LIMITS["J3_MAX"])
    joint_deg[3] = clamp(joint_deg[3], JOINT_LIMITS["J4_MIN"], JOINT_LIMITS["J4_MAX"])
    joint_deg[4] = clamp(joint_deg[4], JOINT_LIMITS["J5_MIN"], JOINT_LIMITS["J5_MAX"])
    joint_deg[5] = clamp(joint_deg[5], JOINT_LIMITS["J6_MIN"], JOINT_LIMITS["J6_MAX"])
    return joint_deg

def preprocess_image_bgr_to_rgb_chw(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return img

def next_episode_id(root: Path) -> str:
    ensure_dir(root)
    existing = sorted([p.name for p in root.glob("episode_*") if p.is_dir()])
    if not existing:
        return "episode_000001"
    last_num = int(existing[-1].split("_")[-1])
    return f"episode_{last_num + 1:06d}"

def load_camera_config(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing camera config: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg

def print_camera_config(cfg, label):
    print(f"[INFO] {label} config: {cfg['resolution']['width']}x{cfg['resolution']['height']}@{cfg['fps']}")


# =========================================================
# 3. RealSense (异步无阻塞多线程版本)
# =========================================================
class RealSenseReader:
    def __init__(self, serial, width=640, height=480, fps=30, camera_name="camera", config_dict=None):
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_name = camera_name
        self.config_dict = config_dict or {}

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.profile = None
        self.latest_color = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        self.profile = self.pipeline.start(self.config)
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        print(f"[INFO] {self.camera_name} camera async reader thread started.")

    def _read_loop(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=500)
                color_frame = frames.get_color_frame()
                if color_frame:
                    img = np.asanyarray(color_frame.get_data())
                    with self.lock:
                        self.latest_color = img
            except RuntimeError:
                continue
            except Exception as e:
                print(f"[WARN] {self.camera_name} camera read error: {e}")

    def stop(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        try:
            self.pipeline.stop()
        except Exception:
            pass
        print(f"[INFO] {self.camera_name} camera stopped.")

    def get_color(self):
        with self.lock:
            if self.latest_color is None:
                return None
            return self.latest_color.copy()


# =========================================================
# 4. Robot Interface
# =========================================================
class XArmRolloutInterface:
    def __init__(self, ip):
        self.arm = XArmAPI(ip)

    def connect(self):
        self.arm.clean_error()
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(1)  
        self.arm.set_state(0)
        time.sleep(0.5)

        try:
            self.arm.set_gripper_enable(True)
            self.arm.set_gripper_mode(0)
        except Exception as e:
            print(f"[WARN] Gripper init warning: {e}")

    def disconnect(self):
        try: self.arm.disconnect()
        except Exception: pass

    def get_state(self):
        code_j, joints = self.arm.get_servo_angle(is_radian=False)
        code_p, tcp = self.arm.get_position(is_radian=False)
        if code_j != 0 or code_p != 0:
            raise RuntimeError(f"Failed to read robot state: code_j={code_j}, code_p={code_p}")

        try:
            code_g, grip_pos = self.arm.get_gripper_position()
            if code_g != 0: grip_pos = 0.0
        except Exception:
            grip_pos = 0.0

        return {
            "joint": np.array(joints[:6], dtype=np.float32),
            "tcp": np.array(tcp[:6], dtype=np.float32),
            "gripper_pos": np.float32(grip_pos),
        }

    def send_joint(self, joint_target_deg):
        try:
            code = self.arm.set_servo_angle_j(angles=[float(x) for x in joint_target_deg], is_radian=False)
            return 0 if code == 0 else code
        except Exception:
            return -1

    def send_gripper(self, cmd_gripper):
        if not ENABLE_GRIPPER: return 0
        try:
            code = self.arm.set_gripper_position(float(cmd_gripper), wait=False)
            return 0 if code is None else code
        except Exception:
            return -1

    def get_err_warn_code(self):
        try: return self.arm.get_err_warn_code()
        except Exception: return None

    def emergency_stop(self):
        try: self.arm.emergency_stop()
        except Exception: pass

    def move_to_home(self):
        print("[INFO] Returning robot to home pose...")
        try:
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(0.05)
            self.arm.set_servo_angle(angle=HOME_JOINT, speed=RESET_SPEED, mvacc=RESET_ACC, wait=True, is_radian=False)
        except Exception as e:
            print(f"[WARN] move_to_home failed: {e}")
        finally:
            try:
                self.arm.set_mode(1)
                self.arm.set_state(0)
                time.sleep(0.05)
            except Exception:
                pass
        print("[INFO] Home reached.")


# =========================================================
# 异步高频伺服插值线程 (引入三次样条与惯性滑行容错)
# =========================================================
class AsyncServoController:
    def __init__(self, robot_interface, servo_hz=100.0, control_hz=10.0):
        self.robot = robot_interface
        self.dt = 1.0 / servo_hz
        self.control_dt = 1.0 / control_hz
        self.steps_per_interval = int(servo_hz / control_hz)
        
        self.running = False
        self.lock = threading.Lock()
        self.thread = None 
        
        self.current_joint = None
        self.current_vel = None
        self.current_gripper = None
        self.target_gripper = None
        
        self.trajectory_buffer = []

    def start(self, init_joint, init_gripper):
        self.current_joint = np.array(init_joint, dtype=np.float64)
        self.current_vel = np.zeros(6, dtype=np.float64)
        
        self.current_gripper = float(init_gripper)
        self.target_gripper = self.current_gripper
        
        self.trajectory_buffer = []
        self.running = True
        
        self.thread = threading.Thread(target=self._servo_loop, daemon=True)
        self.thread.start()
        print(f"[INFO] Async Servo Controller started at {int(1.0/self.dt)}Hz")

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        print("[INFO] Async Servo Controller stopped.")

    def update_target(self, target_joint, target_gripper):
        with self.lock:
            q0 = self.current_joint.copy()
            v0 = self.current_vel.copy()
            q1 = np.array(target_joint, dtype=np.float64)
            
            # 估算 0.1s 后的终点速度，保持连续运动
            v1 = (q1 - q0) / self.control_dt
            
            self.trajectory_buffer = []
            
            # 生成 0.1s 内的三次多项式(Hermite Spline)轨迹点
            for i in range(1, self.steps_per_interval + 1):
                t = i * self.dt
                
                a = (v0 + v1) / (self.control_dt**2) - 2 * (q1 - q0) / (self.control_dt**3)
                b = 3 * (q1 - q0) / (self.control_dt**2) - (2 * v0 + v1) / self.control_dt
                c = v0
                d = q0
                
                qt = a * (t**3) + b * (t**2) + c * t + d
                vt = 3 * a * (t**2) + 2 * b * t + c
                
                self.trajectory_buffer.append((qt, vt))
                
            self.target_gripper = float(target_gripper)

    def _servo_loop(self):
        while self.running:
            loop_t0 = time.time()
            
            with self.lock:
                if len(self.trajectory_buffer) > 0:
                    tgt_j, tgt_v = self.trajectory_buffer.pop(0)
                    self.current_joint = tgt_j
                    self.current_vel = tgt_v
                else:
                    # 惯性滑行：如果推理由于抖动慢了，保持平滑移动不急停
                    self.current_joint += self.current_vel * self.dt
                    # 轻微衰减防止失控飞走
                    self.current_vel *= 0.95 
                
                tgt_g = self.target_gripper
            
            self.robot.send_joint(self.current_joint)
            
            if tgt_g is not None and abs(tgt_g - self.current_gripper) > 5.0: 
                self.robot.send_gripper(tgt_g)
                self.current_gripper = tgt_g
                
            elapsed = time.time() - loop_t0
            time.sleep(max(0.0, self.dt - elapsed))


# =========================================================
# 5. Recorder & Post-processing Utils
# =========================================================
class RolloutRecorder:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        ensure_dir(self.root_dir)
        self.active = False

    def start(self, ckpt_dir: str, front_cfg_path: str, wrist_cfg_path: str):
        if self.active: return False
        self.episode_id = next_episode_id(self.root_dir)
        self.episode_dir = self.root_dir / self.episode_id
        ensure_dir(self.episode_dir)

        self.front_dir = self.episode_dir / "front_rgb"
        self.wrist_dir = self.episode_dir / "wrist_rgb"
        ensure_dir(self.front_dir)
        ensure_dir(self.wrist_dir)

        self.state_file = open(self.episode_dir / "robot_state.csv", "w", newline="", encoding="utf-8")
        self.state_writer = csv.DictWriter(self.state_file, fieldnames=["timestamp_ns", "j1", "j2", "j3", "j4", "j5", "j6", "tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz", "gripper_pos"])
        self.state_writer.writeheader()

        self.pred_file = open(self.episode_dir / "pred_chunks.csv", "w", newline="", encoding="utf-8")
        pred_fields = ["timestamp_ns", "step_idx"]
        for k in range(CHUNK_SIZE): pred_fields += [f"pred_{k}_j1", f"pred_{k}_j2", f"pred_{k}_j3", f"pred_{k}_j4", f"pred_{k}_j5", f"pred_{k}_j6", f"pred_{k}_gripper"]
        self.pred_writer = csv.DictWriter(self.pred_file, fieldnames=pred_fields)
        self.pred_writer.writeheader()

        self.action_file = open(self.episode_dir / "executed_action.csv", "w", newline="", encoding="utf-8")
        self.action_writer = csv.DictWriter(self.action_file, fieldnames=["timestamp_ns", "step_idx", "exec_j1", "exec_j2", "exec_j3", "exec_j4", "exec_j5", "exec_j6", "exec_gripper"])
        self.action_writer.writeheader()

        self.frame_idx = 0
        self.active = True
        return True

    def write_step(self, front_bgr, wrist_bgr, state, pred_chunk, exec_joint, exec_gripper, step_idx):
        if not self.active: return
        ts = now_ns()
        cv2.imwrite(str(self.front_dir / f"{self.frame_idx:06d}.png"), front_bgr)
        cv2.imwrite(str(self.wrist_dir / f"{self.frame_idx:06d}.png"), wrist_bgr)

        self.state_writer.writerow({"timestamp_ns": ts, "j1": state["joint"][0], "j2": state["joint"][1], "j3": state["joint"][2], "j4": state["joint"][3], "j5": state["joint"][4], "j6": state["joint"][5], "tcp_x": state["tcp"][0], "tcp_y": state["tcp"][1], "tcp_z": state["tcp"][2], "tcp_rx": state["tcp"][3], "tcp_ry": state["tcp"][4], "tcp_rz": state["tcp"][5], "gripper_pos": state["gripper_pos"]})

        pred_row = {"timestamp_ns": ts, "step_idx": step_idx}
        for k in range(CHUNK_SIZE):
            pred_row.update({f"pred_{k}_j1": pred_chunk[k,0], f"pred_{k}_j2": pred_chunk[k,1], f"pred_{k}_j3": pred_chunk[k,2], f"pred_{k}_j4": pred_chunk[k,3], f"pred_{k}_j5": pred_chunk[k,4], f"pred_{k}_j6": pred_chunk[k,5], f"pred_{k}_gripper": pred_chunk[k,6]})
        self.pred_writer.writerow(pred_row)

        self.action_writer.writerow({"timestamp_ns": ts, "step_idx": step_idx, "exec_j1": exec_joint[0], "exec_j2": exec_joint[1], "exec_j3": exec_joint[2], "exec_j4": exec_joint[3], "exec_j5": exec_joint[4], "exec_j6": exec_joint[5], "exec_gripper": exec_gripper})
        self.frame_idx += 1

    def stop(self, success: bool, reason: str = None):
        if not self.active: return False
        self.state_file.close()
        self.pred_file.close()
        self.action_file.close()
        self.active = False
        return True

def load_official_act_policy(ckpt_dir, ckpt_name, device):
    ckpt_dir = Path(ckpt_dir)
    stats_path = ckpt_dir / "dataset_stats.pkl"
    with open(stats_path, "rb") as f: stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]

    policy_config = {"lr": 1e-5, "num_queries": CHUNK_SIZE, "kl_weight": 10, "hidden_dim": 512, "dim_feedforward": 3200, "lr_backbone": 1e-5, "backbone": "resnet18", "enc_layers": 4, "dec_layers": 7, "nheads": 8, "camera_names": CAMERA_NAMES, "state_dim": STATE_DIM, "action_dim": ACTION_DIM}
    policy = ACTPolicy(policy_config)
    policy.load_state_dict(torch.load(ckpt_dir / ckpt_name, map_location=device))
    policy.to(device)
    policy.eval()
    return policy, pre_process, post_process

def postprocess_action(pred_action, current_joint):
    pred_joint = np.asarray(pred_action[:6], dtype=np.float32).copy()
    pred_gripper = float(pred_action[6])

    delta_joint = pred_joint - current_joint
    delta_joint = np.clip(delta_joint, -MAX_JOINT_STEP_DEG, MAX_JOINT_STEP_DEG)
    target_joint = clamp_joint(current_joint + delta_joint)

    cmd_gripper = float(np.clip(pred_gripper, GRIPPER_MIN, GRIPPER_MAX))
    return target_joint, cmd_gripper, pred_gripper


# =========================================================
# 8. Main rollout
# =========================================================
@torch.no_grad()
def rollout():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    front_cfg = load_camera_config(FRONT_CAM_CONFIG_PATH)
    wrist_cfg = load_camera_config(WRIST_CAM_CONFIG_PATH)

    policy, pre_process, post_process = load_official_act_policy(CKPT_DIR, CKPT_NAME, device)

    robot = XArmRolloutInterface(IP_LEFT)
    robot.connect()
    print("[INFO] Robot connected.")
    
    servo_controller = AsyncServoController(robot, servo_hz=100.0, control_hz=CONTROL_HZ)

    front_cam = RealSenseReader(serial=FRONT_CAM_SERIAL, width=front_cfg["resolution"]["width"], height=front_cfg["resolution"]["height"], fps=front_cfg["fps"], camera_name="front", config_dict=front_cfg)
    wrist_cam = RealSenseReader(serial=WRIST_CAM_SERIAL, width=wrist_cfg["resolution"]["width"], height=wrist_cfg["resolution"]["height"], fps=wrist_cfg["fps"], camera_name="wrist", config_dict=wrist_cfg)

    front_cam.start()
    wrist_cam.start()
    recorder = RolloutRecorder(ROLLOUT_ROOT)

    dt = 1.0 / CONTROL_HZ
    step_idx = 0
    paused = True
    rollout_active = False

    if TEMPORAL_AGG:
        all_time_actions = torch.zeros([1000, 1000 + CHUNK_SIZE, ACTION_DIM], device=device)

    try:
        # 绝对时钟初始化
        next_step_time = time.time()
        
        while True:
            front_bgr = front_cam.get_color()
            wrist_bgr = wrist_cam.get_color()
            if front_bgr is None or wrist_bgr is None: continue

            state = robot.get_state()
            joint = state["joint"]
            tcp = state["tcp"]
            gripper_pos = state["gripper_pos"]

            vis_front = front_bgr.copy()
            vis_wrist = wrist_bgr.copy()
            cv2.putText(vis_front, f"step={step_idx}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Official ACT xArm Rollout", np.hstack([vis_front, vis_wrist]))
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"): break
            elif key == ord("e"): robot.emergency_stop(); break
            elif key == ord("h"):
                paused = True
                if not rollout_active: robot.move_to_home()
            elif key == ord("r"):
                if not rollout_active:
                    step_idx = 0
                    paused = False
                    rollout_active = True
                    recorder.start(ckpt_dir=CKPT_DIR, front_cfg_path=FRONT_CAM_CONFIG_PATH, wrist_cfg_path=WRIST_CAM_CONFIG_PATH)
                    if TEMPORAL_AGG: all_time_actions.zero_()
                    servo_controller.start(state["joint"], state["gripper_pos"])
                    next_step_time = time.time() # 启动时重置绝对时钟
            elif key == ord("s"):
                if rollout_active:
                    servo_controller.stop() 
                    recorder.stop(success=True, reason=None)
                    rollout_active = False
                    paused = True
                    if AUTO_RETURN_HOME_AFTER_STOP: robot.move_to_home()
            elif key == ord("f"):
                if rollout_active:
                    servo_controller.stop() 
                    recorder.stop(success=False, reason="manual_failure")
                    rollout_active = False
                    paused = True
                    if AUTO_RETURN_HOME_AFTER_STOP: robot.move_to_home()
            elif key == ord(" "):
                if rollout_active: paused = not paused

            if not rollout_active or paused:
                time.sleep(0.02)
                next_step_time = time.time() # 暂停期间保持时钟同步，防止解除暂停时暴走
                continue

            # ----------------------------------------------------
            # 模型推理部分
            # ----------------------------------------------------
            image = np.stack([preprocess_image_bgr_to_rgb_chw(front_bgr), preprocess_image_bgr_to_rgb_chw(wrist_bgr)], axis=0)
            image = torch.from_numpy(image).float().to(device).unsqueeze(0)

            qpos_numpy = pre_process(np.concatenate([joint, np.array([gripper_pos], dtype=np.float32)], axis=0))
            qpos = torch.from_numpy(qpos_numpy).float().to(device).unsqueeze(0)

            # 逻辑嵌套修复，确保时序正确
            if step_idx % QUERY_FREQUENCY == 0:
                all_actions = policy(qpos, image)  
                if TEMPORAL_AGG:
                    all_time_actions[[step_idx], step_idx:step_idx + CHUNK_SIZE] = all_actions

            if TEMPORAL_AGG:
                actions_for_curr_step = all_time_actions[:, step_idx]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]

                if len(actions_for_curr_step) == 0:
                    raw_action = all_actions[:, 0]
                else:
                    k = 0.01
                    # 权重反转修复，让最新预测占主导
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step))[::-1])
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).float().to(device).unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)

                pred_chunk = all_actions[0].detach().cpu().numpy()
            else:
                raw_action = all_actions[:, step_idx % CHUNK_SIZE]
                pred_chunk = all_actions[0].detach().cpu().numpy()

            raw_action = raw_action.squeeze(0).detach().cpu().numpy()
            pred_action = post_process(raw_action)
            pred_chunk = post_process(pred_chunk)

            exec_joint, exec_gripper, _ = postprocess_action(pred_action, joint)

            # ----------------------------------------------------
            # 控制下发部分 (无阻塞，移交后台三次样条)
            # ----------------------------------------------------
            servo_controller.update_target(exec_joint, exec_gripper)

            recorder.write_step(front_bgr, wrist_bgr, state, pred_chunk, exec_joint, exec_gripper, step_idx)
            step_idx += 1

            # ----------------------------------------------------
            # 绝对时钟对齐，消除系统累积漂移误差
            # ----------------------------------------------------
            next_step_time += dt
            sleep_time = next_step_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 若 GPU 偶尔卡顿超时，重置目标时钟，防止下一帧疯狂快进
                next_step_time = time.time() 

    finally:
        servo_controller.stop()
        if recorder.active: recorder.stop(success=False, reason="interrupted")
        front_cam.stop()
        wrist_cam.stop()
        robot.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    rollout()