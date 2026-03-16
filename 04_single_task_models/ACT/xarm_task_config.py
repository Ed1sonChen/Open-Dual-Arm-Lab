from pathlib import Path

# Edit these paths for your machine.
DATA_ROOT = Path("./dataset/act_hdf5")

# Local task configs for non-ALOHA real robot training.
# This avoids importing aloha_scripts.constants.TASK_CONFIGS.
LOCAL_TASK_CONFIGS = {
    "xarm_flip_block": {
        "dataset_dir": str(DATA_ROOT / "xarm_flip_block"),
        "num_episodes": 30,
        "episode_len": 100,          # padded / nominal episode length in HDF5
        "camera_names": ["front", "wrist"],
        "state_dim": 7,              # [j1..j6, gripper_pos]
        "action_dim": 7,             # [tcp_x, tcp_y, tcp_z, tcp_rx, tcp_ry, tcp_rz, gripper]
        "is_sim": False,
    },
}