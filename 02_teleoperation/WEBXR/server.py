import asyncio
import websockets
import json
import time
import sys
import threading
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI
 
# ==========================================
# 1. configuration
# ==========================================
 
# change to your own robtic arms' IP address
IP_LEFT = ''
IP_RIGHT = ''
SCALE = 500.0
 
LIMITS = {
    'X_MIN': 150, 'X_MAX': 600,
    'Y_MIN': -500, 'Y_MAX': 500,
    'Z_MIN': 0, 'Z_MAX': 600
}
 
# Per-packet motion limits
MAX_STEP_TRANSLATION_MM = 8.0  # maximum translation per packet (mm)
MAX_STEP_ROTATION_DEG = 5.0    # maximum rotation per packet (deg)
 
# Watchdog / timeout
PACKET_TIMEOUT_SEC = 0.20       # if no packet arrives within this time, servo pauses
 
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
            diff = (x - self.x_prev)
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
# 3. Helper functions: motion clamping
# ==========================================
def clamp_vector_norm(vec, max_norm):
    """Clamp vector magnitude to max_norm"""
    norm = np.linalg.norm(vec)
    if norm <= max_norm or norm < 1e-9:
        return vec
    return vec * (max_norm / norm)
 
def clamp_rotation_delta(delta_rot: R, max_deg: float) -> R:
    """
    Clamp incremental rotation to max_deg.
    Uses rotation vector representation for stable limiting.
    """
    rotvec = delta_rot.as_rotvec()
    angle = np.linalg.norm(rotvec)
    max_rad = np.deg2rad(max_deg)
 
    if angle <= max_rad or angle < 1e-9:
        return delta_rot
 
    rotvec_clamped = rotvec * (max_rad / angle)
    return R.from_rotvec(rotvec_clamped)
 
# ==========================================
# 4. Initialization
# ==========================================
 
# Transformation from VR frame to robot frame
M_mat = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0]
])
M_transform = R.from_matrix(M_mat)
M_transform_inv = M_transform.inv()
 
print("Connecting to  xArm...")
try:
    arm_left = XArmAPI(IP_LEFT)
    arm_right = XArmAPI(IP_RIGHT)
except Exception as e:
    print(f"Robot connection failed: {e}")
    sys.exit()
 
for arm in [arm_left, arm_right]:
    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(1)
    arm.set_state(0)
    time.sleep(0.5)
 
print("✅ Dual-arm initialization successful. 1€ Filter servo mode active.")
 
prev_vr_pos = {'left': None, 'right': None}
prev_vr_quat = {'left': None, 'right': None}
last_gripper_pos = {'left': -1, 'right': -1}
 
raw_target_xyz = {'left': None, 'right': None}
raw_target_rpy = {'left': None, 'right': None}
 
# Last time a packet was received
last_packet_time = {'left': 0.0, 'right': 0.0}
 
# Used to avoid printing timeout repeatedly
timeout_triggered = {'left': False, 'right': False}
 
data_lock = threading.Lock()
 
filters = {
    'left_pos': OneEuroFilter3D(mincutoff=0.8, beta=0.04),
    'left_rot': OneEuroFilter3D(mincutoff=0.5, beta=0.02),
    'right_pos': OneEuroFilter3D(mincutoff=0.8, beta=0.04),
    'right_rot': OneEuroFilter3D(mincutoff=0.5, beta=0.02)
}
 
# ==========================================
# 5. State reset utilities
# ==========================================
def reset_hand_state(handedness):
    global prev_vr_pos, prev_vr_quat, raw_target_xyz, raw_target_rpy
    global last_packet_time, timeout_triggered
 
    with data_lock:
        prev_vr_pos[handedness] = None
        prev_vr_quat[handedness] = None
        raw_target_xyz[handedness] = None
        raw_target_rpy[handedness] = None
        last_packet_time[handedness] = 0.0
        timeout_triggered[handedness] = False
 
    filters[f'{handedness}_pos'].reset()
    filters[f'{handedness}_rot'].reset()
 
def reset_all_states():
    reset_hand_state('left')
    reset_hand_state('right')
 
# ==========================================
# 6. High-frequency servo control loop
# ==========================================
def robot_control_loop(handedness, arm):
    global raw_target_xyz, raw_target_rpy, last_packet_time, timeout_triggered
 
    code, current_tcp = arm.get_position()
    if code != 0:
        print(f"❌ {handedness} arm get_position failed, code={code}")
        return
 
    while True:
        now = time.time()
 
        with data_lock:
            t_xyz = raw_target_xyz[handedness]
            t_rpy = raw_target_rpy[handedness]
            pkt_time = last_packet_time[handedness]
 
        # Watchdog: stop servo if packet stream stops
        if pkt_time > 0 and (now - pkt_time) > PACKET_TIMEOUT_SEC:
            if not timeout_triggered[handedness]:
                print(f"⚠️ {handedness} hand timeout: no packet for {now - pkt_time:.3f}s, servo paused.")
                timeout_triggered[handedness] = True
 
            with data_lock:
                prev_vr_pos[handedness] = None
                prev_vr_quat[handedness] = None
                raw_target_xyz[handedness] = None
                raw_target_rpy[handedness] = None
 
            filters[f'{handedness}_pos'].reset()
            filters[f'{handedness}_rot'].reset()
 
            time.sleep(0.01)
            continue
 
        if pkt_time > 0 and timeout_triggered[handedness]:
            timeout_triggered[handedness] = False
            print(f"✅ {handedness} hand packet stream recovered.")
 
        if t_xyz is not None and t_rpy is not None:
            f_xyz = filters[f'{handedness}_pos'](np.array(t_xyz), is_angle=False)
            f_rpy = filters[f'{handedness}_rot'](np.array(t_rpy), is_angle=True)
 
            final_target = [
                float(f_xyz[0]), float(f_xyz[1]), float(f_xyz[2]),
                float(f_rpy[0]), float(f_rpy[1]), float(f_rpy[2])
            ]
            arm.set_servo_cartesian(final_target)
 
        time.sleep(0.01)
 
threading.Thread(target=robot_control_loop, args=('left', arm_left), daemon=True).start()
threading.Thread(target=robot_control_loop, args=('right', arm_right), daemon=True).start()
 
# ==========================================
# 7. Process VR input packets
# ==========================================
def process_arm_movement(handedness, vr_data, arm):
    global prev_vr_pos, prev_vr_quat, raw_target_xyz, raw_target_rpy
    global last_gripper_pos, last_packet_time
 
    vr_pos = vr_data['position']
    vr_q = vr_data['quaternion']
    trigger = vr_data['trigger']
    grip = vr_data['grip']
 
     # Update packet timestamp
    with data_lock:
        last_packet_time[handedness] = time.time()
 
    # -----------------------------
    # Gripper control
    # -----------------------------
    gripper_target = int(850 - (trigger * 850))
    if abs(gripper_target - last_gripper_pos[handedness]) > 15:
        arm.set_gripper_position(gripper_target, wait=False)
        last_gripper_pos[handedness] = gripper_target
 
    with data_lock:
        if grip > 0.5:
            curr_vr_quat = R.from_quat([
                vr_q['x'], vr_q['y'], vr_q['z'], vr_q['w']
            ])
 
            # First grip press: initialize reference
            if prev_vr_pos[handedness] is None:
                code, current_tcp = arm.get_position()
                if code == 0:
                    raw_target_xyz[handedness] = [
                        current_tcp[0], current_tcp[1], current_tcp[2]
                    ]
                    raw_target_rpy[handedness] = [
                        current_tcp[3], current_tcp[4], current_tcp[5]
                    ]
                prev_vr_pos[handedness] = vr_pos
                prev_vr_quat[handedness] = curr_vr_quat
                return
 
            # -----------------------------
            # Compute VR motion delta
            # -----------------------------
            dx_vr = vr_pos['x'] - prev_vr_pos[handedness]['x']
            dy_vr = vr_pos['y'] - prev_vr_pos[handedness]['y']
            dz_vr = vr_pos['z'] - prev_vr_pos[handedness]['z']
 
            delta_xyz_robot = np.array([
                -dz_vr * SCALE,
                -dx_vr * SCALE,
                +dy_vr * SCALE
            ], dtype=float)
 
            delta_xyz_robot = clamp_vector_norm(
                delta_xyz_robot,
                MAX_STEP_TRANSLATION_MM
            )
 
            new_xyz = np.array(raw_target_xyz[handedness], dtype=float) + delta_xyz_robot
 
            new_xyz[0] = max(LIMITS['X_MIN'], min(new_xyz[0], LIMITS['X_MAX']))
            new_xyz[1] = max(LIMITS['Y_MIN'], min(new_xyz[1], LIMITS['Y_MAX']))
            new_xyz[2] = max(LIMITS['Z_MIN'], min(new_xyz[2], LIMITS['Z_MAX']))
 
            raw_target_xyz[handedness] = new_xyz.tolist()
 
            # -----------------------------
            # # Rotation delta
            # -----------------------------
            delta_R_vr = curr_vr_quat * prev_vr_quat[handedness].inv()
            delta_R_rob = M_transform * delta_R_vr * M_transform_inv
            delta_R_rob = clamp_rotation_delta(
                delta_R_rob,
                MAX_STEP_ROTATION_DEG
            )
 
            current_R_rob = R.from_euler(
                'xyz',
                raw_target_rpy[handedness],
                degrees=True
            )
            new_R_rob = delta_R_rob * current_R_rob
 
            raw_target_rpy[handedness] = new_R_rob.as_euler(
                'xyz',
                degrees=True
            ).tolist()
 
            prev_vr_pos[handedness] = vr_pos
            prev_vr_quat[handedness] = curr_vr_quat
 
        else:
            prev_vr_pos[handedness] = None
            prev_vr_quat[handedness] = None
            raw_target_xyz[handedness] = None
            raw_target_rpy[handedness] = None
 
            filters[f'{handedness}_pos'].reset()
            filters[f'{handedness}_rot'].reset()
 
# ==========================================
# 8. WebSocket server
# ==========================================
async def handle_vr_data(websocket):
    print("🚀 Quest 3 teleoperation session start!")
    try:
        async for message in websocket:
            data = json.loads(message)
 
            if 'left' in data:
                process_arm_movement('left', data['left'], arm_left)
 
            if 'right' in data:
                process_arm_movement('right', data['right'], arm_right)
 
    except websockets.exceptions.ConnectionClosed:
        print("❌ Quest 3 connection closed")
        reset_all_states()
        arm_left.set_state(4)
        arm_right.set_state(4)
 
    except Exception as e:
        print(f"❌ WebSocket handler error: {e}")
        reset_all_states()
        arm_left.set_state(4)
        arm_right.set_state(4)
 
async def main():
    async with websockets.serve(handle_vr_data, "0.0.0.0", 8765):
        print("Waiting for VR client connection...")
        await asyncio.Future()
 
if __name__ == "__main__":
    asyncio.run(main())