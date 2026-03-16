#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd


STATE_COLS = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper_pos"]

# Use next-step joint state as ACT action.
ACTION_COLS = ["j1", "j2", "j3", "j4", "j5", "j6", "cmd_gripper"]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def compute_qvel(qpos: np.ndarray, dt: float) -> np.ndarray:
    """
    qpos: [T, D]
    qvel: finite difference velocity in units per second
    """
    qvel = np.zeros_like(qpos, dtype=np.float32)
    if len(qpos) > 1:
        qvel[1:] = (qpos[1:] - qpos[:-1]) / dt
        qvel[0] = qvel[1]
    return qvel


def is_success_episode(ep_dir: Path) -> bool:
    meta_path = ep_dir / "meta.json"
    if not meta_path.exists():
        print(f"[SKIP] {ep_dir.name}: missing meta.json")
        return False

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        print(f"[SKIP] {ep_dir.name}: failed to read meta.json ({e})")
        return False

    success = meta.get("raw_meta", {}).get("success", False)
    if not success:
        failure_reason = meta.get("raw_meta", {}).get("failure_reason", "unknown")
        print(f"[SKIP] {ep_dir.name}: success=False, reason={failure_reason}")
        return False

    return True


def trim_idle_prefix_suffix(
    df: pd.DataFrame,
    joint_eps_deg: float = 0.25,
    gripper_eps: float = 5.0,
    min_keep: int = 8,
):
    """
    Remove long idle prefix/suffix where qpos barely changes.
    Helps ACT avoid learning large dead zones at the beginning/end.
    """
    q = df[STATE_COLS].to_numpy(dtype=np.float32)
    if len(q) < min_keep:
        return df.copy()

    dq = np.zeros_like(q, dtype=np.float32)
    dq[1:] = np.abs(q[1:] - q[:-1])
    dq[0] = dq[1] if len(q) > 1 else 0.0

    moving = (
        (dq[:, :6].max(axis=1) > joint_eps_deg) |
        (dq[:, 6] > gripper_eps)
    )

    valid_idx = np.where(moving)[0]
    if len(valid_idx) == 0:
        return df.copy()

    start = max(0, int(valid_idx[0]) - 2)
    end = min(len(df) - 1, int(valid_idx[-1]) + 2)

    trimmed = df.iloc[start:end + 1].reset_index(drop=True)
    if len(trimmed) < min_keep:
        return df.copy()
    return trimmed


# def build_joint_action(df: pd.DataFrame) -> np.ndarray:
#     """
#     action[t] = qpos[t+1]
#     For the final frame, repeat the last valid action.
#     """
#     q = df[ACTION_COLS].to_numpy(dtype=np.float32)   # [T,7]
#     action = np.zeros_like(q, dtype=np.float32)

#     if len(q) == 1:
#         action[0] = q[0]
#         return action

#     action[:-1] = q[1:]
#     action[-1] = q[-1]
#     return action

def build_joint_action(df: pd.DataFrame) -> np.ndarray:
    q_joints = df[["j1", "j2", "j3", "j4", "j5", "j6"]].to_numpy(dtype=np.float32)
    q_grip = df["cmd_gripper"].to_numpy(dtype=np.float32)

    T = len(df)
    action = np.zeros((T, 7), dtype=np.float32)

    if T == 1:
        action[0, :6] = q_joints[0]
        action[0, 6] = q_grip[0]
        return action

    action[:-1, :6] = q_joints[1:]
    action[-1, :6] = q_joints[-1]

    action[:, 6] = q_grip

    return action


def maybe_normalize_gripper(
    qpos: np.ndarray,
    action: np.ndarray,
    gripper_open: float = None,
    gripper_close: float = None,
):
    """
    Optional normalization of gripper from raw xArm units to [0, 1].
    0 = closed, 1 = open
    """
    if gripper_open is None or gripper_close is None:
        return qpos, action

    denom = float(gripper_open - gripper_close)
    if abs(denom) < 1e-6:
        raise ValueError("Invalid gripper normalization range: open and close are too close.")

    def normalize(x):
        y = (x - gripper_close) / denom
        return np.clip(y, 0.0, 1.0)

    qpos = qpos.copy()
    action = action.copy()
    qpos[:, -1] = normalize(qpos[:, -1])
    action[:, -1] = normalize(action[:, -1])
    return qpos, action


def convert_episode(
    ep_dir: Path,
    out_dir: Path,
    episode_idx: int,
    trim_idle: bool = True,
    joint_eps_deg: float = 0.25,
    gripper_eps: float = 5.0,
    gripper_open: float = None,
    gripper_close: float = None,
):
    samples_path = ep_dir / "samples.csv"
    meta_path = ep_dir / "meta.json"

    if not samples_path.exists():
        raise FileNotFoundError(f"Missing {samples_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    df = pd.read_csv(samples_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    target_hz = float(meta.get("target_hz", 10.0))
    dt = 1.0 / target_hz

    if trim_idle:
        df = trim_idle_prefix_suffix(
            df,
            joint_eps_deg=joint_eps_deg,
            gripper_eps=gripper_eps,
        )

    T = len(df)
    if T < 2:
        raise RuntimeError(f"Too few samples after trimming in {ep_dir.name}: T={T}")

    qpos = df[STATE_COLS].to_numpy(dtype=np.float32)   # [T,7]
    action = build_joint_action(df)                    # [T,7]
    qvel = compute_qvel(qpos, dt)

    qpos, action = maybe_normalize_gripper(
        qpos=qpos,
        action=action,
        gripper_open=gripper_open,
        gripper_close=gripper_close,
    )

    front_imgs = []
    wrist_imgs = []

    for _, row in df.iterrows():
        front_path = Path(row["front_rgb"])
        wrist_path = Path(row["wrist_rgb"])

        if not front_path.is_absolute():
            front_path = ep_dir / front_path
        if not wrist_path.is_absolute():
            wrist_path = ep_dir / wrist_path

        front_imgs.append(load_rgb(front_path))
        wrist_imgs.append(load_rgb(wrist_path))

    front_imgs = np.stack(front_imgs, axis=0)   # [T,H,W,3]
    wrist_imgs = np.stack(wrist_imgs, axis=0)   # [T,H,W,3]

    ensure_dir(out_dir)
    out_path = out_dir / f"episode_{episode_idx}.hdf5"

    with h5py.File(out_path, "w") as f:
        f.attrs["sim"] = False
        f.attrs["source_episode"] = ep_dir.name
        f.attrs["target_hz"] = target_hz
        f.attrs["action_type"] = "next_step_joint_state"
        f.attrs["state_cols"] = json.dumps(STATE_COLS)
        f.attrs["action_cols"] = json.dumps(ACTION_COLS)
        f.attrs["trim_idle"] = bool(trim_idle)

        if gripper_open is not None and gripper_close is not None:
            f.attrs["gripper_normalized"] = True
            f.attrs["gripper_open"] = float(gripper_open)
            f.attrs["gripper_close"] = float(gripper_close)
        else:
            f.attrs["gripper_normalized"] = False

        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=qpos, dtype=np.float32)
        obs.create_dataset("qvel", data=qvel, dtype=np.float32)

        images = obs.create_group("images")
        images.create_dataset("front", data=front_imgs, dtype=np.uint8)
        images.create_dataset("wrist", data=wrist_imgs, dtype=np.uint8)

        f.create_dataset("action", data=action, dtype=np.float32)

    print(
        f"[OK] wrote {out_path} | source={ep_dir.name} | "
        f"T={T} | action=next_step_joint_state"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--episode_filter_json",
        type=str,
        default=None,
        help='Optional JSON list of episode directory names, e.g. ["episode_000001","episode_000002"]'
    )
    parser.add_argument(
        "--no_trim_idle",
        action="store_true",
        help="Disable trimming idle prefix/suffix."
    )
    parser.add_argument("--joint_eps_deg", type=float, default=0.25)
    parser.add_argument("--gripper_eps", type=float, default=5.0)
    parser.add_argument(
        "--gripper_open",
        type=float,
        default=None,
        help="Optional gripper open value for normalization to [0,1]."
    )
    parser.add_argument(
        "--gripper_close",
        type=float,
        default=None,
        help="Optional gripper close value for normalization to [0,1]."
    )
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    out_dir = Path(args.out_dir)

    ep_dirs = sorted([p for p in processed_root.glob("episode_*") if p.is_dir()])
    if len(ep_dirs) == 0:
        raise RuntimeError(f"No processed episodes found in {processed_root}")

    if args.episode_filter_json is not None:
        keep = set(json.loads(args.episode_filter_json))
        ep_dirs = [p for p in ep_dirs if p.name in keep]

    ep_dirs = [p for p in ep_dirs if is_success_episode(p)]

    if len(ep_dirs) == 0:
        raise RuntimeError("No successful episodes found after filtering.")

    ensure_dir(out_dir)

    for i, ep_dir in enumerate(ep_dirs):
        convert_episode(
            ep_dir=ep_dir,
            out_dir=out_dir,
            episode_idx=i,
            trim_idle=not args.no_trim_idle,
            joint_eps_deg=args.joint_eps_deg,
            gripper_eps=args.gripper_eps,
            gripper_open=args.gripper_open,
            gripper_close=args.gripper_close,
        )

    print(f"[DONE] converted {len(ep_dirs)} successful episodes to {out_dir}")


if __name__ == "__main__":
    main()
