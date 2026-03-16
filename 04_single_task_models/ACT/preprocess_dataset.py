#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def get_episode_dirs(raw_root: Path):
    return sorted([p for p in raw_root.glob("episode_*") if p.is_dir()])


def nearest_row(df: pd.DataFrame, t_ns: int):
    idx = (df["timestamp_ns"] - t_ns).abs().idxmin()
    return df.loc[idx]


def parse_image_timestamp_ns(path: Path) -> int:
    """
    Parse timestamp from image filename.

    Expected filename format:
        000123_1734567890123456789.png
    """
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid image filename format: {path.name}")
    return int(parts[-1])


def build_image_timestamp_array(image_files):
    ts = []
    for p in image_files:
        ts.append(parse_image_timestamp_ns(p))
    return np.array(ts, dtype=np.int64)


def nearest_image_file(image_files, image_ts, t_ns, max_diff_ns=None):
    if len(image_files) == 0:
        return None, None

    idx = int(np.argmin(np.abs(image_ts - t_ns)))
    diff_ns = int(abs(int(image_ts[idx]) - int(t_ns)))

    if max_diff_ns is not None and diff_ns > max_diff_ns:
        return None, diff_ns

    return image_files[idx], diff_ns


def trim_valid_range(mask: np.ndarray):
    """
    Keep only the continuous range between first valid and last valid sample.
    """
    valid_idx = np.where(mask)[0]
    if len(valid_idx) == 0:
        return None, None
    return int(valid_idx[0]), int(valid_idx[-1])


def process_episode(
    ep_dir: Path,
    out_root: Path,
    target_hz: float = 10.0,
    copy_images: bool = False,
    max_state_gap_ms: float = 100.0,
    max_cmd_gap_ms: float = 100.0,
    max_img_gap_ms: float = 100.0,
):
    robot_df = load_csv(ep_dir / "robot_state.csv")
    cmd_df = load_csv(ep_dir / "teleop_command.csv")

    if robot_df is None or len(robot_df) == 0:
        print(f"[WARN] Skip {ep_dir.name}: missing robot_state.csv")
        return False
    if cmd_df is None or len(cmd_df) == 0:
        print(f"[WARN] Skip {ep_dir.name}: missing teleop_command.csv")
        return False

    front_files = sorted((ep_dir / "front_rgb").glob("*.png"))
    wrist_files = sorted((ep_dir / "wrist_rgb").glob("*.png"))

    if len(front_files) == 0 or len(wrist_files) == 0:
        print(f"[WARN] Skip {ep_dir.name}: missing front or wrist images")
        return False

    try:
        front_ts = build_image_timestamp_array(front_files)
        wrist_ts = build_image_timestamp_array(wrist_files)
    except Exception as e:
        print(f"[WARN] Skip {ep_dir.name}: failed to parse image timestamps ({e})")
        return False

    # Global overlap across robot state, command, and both cameras
    t0 = max(
        int(robot_df["timestamp_ns"].min()),
        int(cmd_df["timestamp_ns"].min()),
        int(front_ts.min()),
        int(wrist_ts.min()),
    )
    t1 = min(
        int(robot_df["timestamp_ns"].max()),
        int(cmd_df["timestamp_ns"].max()),
        int(front_ts.max()),
        int(wrist_ts.max()),
    )

    if t1 <= t0:
        print(f"[WARN] Skip {ep_dir.name}: invalid overlap window")
        return False

    step_ns = int(1e9 / target_hz)
    grid = np.arange(t0, t1 + 1, step_ns, dtype=np.int64)

    if len(grid) < 2:
        print(f"[WARN] Skip {ep_dir.name}: too few aligned samples")
        return False

    max_state_gap_ns = int(max_state_gap_ms * 1e6)
    max_cmd_gap_ns = int(max_cmd_gap_ms * 1e6)
    max_img_gap_ns = int(max_img_gap_ms * 1e6)

    out_ep = out_root / ep_dir.name
    ensure_dir(out_ep)

    if copy_images:
        out_front = out_ep / "front_rgb"
        out_wrist = out_ep / "wrist_rgb"
        ensure_dir(out_front)
        ensure_dir(out_wrist)
    else:
        out_front = None
        out_wrist = None

    rows = []
    front_name_map = {}
    wrist_name_map = {}

    skipped_state = 0
    skipped_cmd = 0
    skipped_front = 0
    skipped_wrist = 0

    # Precompute nearest validity so we can optionally trim later
    candidate_rows = []

    for step_idx, t_ns in enumerate(grid):
        r = nearest_row(robot_df, t_ns)
        c = nearest_row(cmd_df, t_ns)

        r_diff_ns = int(abs(int(r["timestamp_ns"]) - int(t_ns)))
        c_diff_ns = int(abs(int(c["timestamp_ns"]) - int(t_ns)))

        front_file, front_diff_ns = nearest_image_file(
            front_files, front_ts, t_ns, max_diff_ns=max_img_gap_ns
        )
        wrist_file, wrist_diff_ns = nearest_image_file(
            wrist_files, wrist_ts, t_ns, max_diff_ns=max_img_gap_ns
        )

        valid = True

        if r_diff_ns > max_state_gap_ns:
            skipped_state += 1
            valid = False
        if c_diff_ns > max_cmd_gap_ns:
            skipped_cmd += 1
            valid = False
        if front_file is None:
            skipped_front += 1
            valid = False
        if wrist_file is None:
            skipped_wrist += 1
            valid = False

        candidate_rows.append({
            "step_idx": step_idx,
            "t_ns": int(t_ns),
            "robot_row": r,
            "cmd_row": c,
            "robot_diff_ns": r_diff_ns,
            "cmd_diff_ns": c_diff_ns,
            "front_file": front_file,
            "wrist_file": wrist_file,
            "front_diff_ns": front_diff_ns,
            "wrist_diff_ns": wrist_diff_ns,
            "valid": valid,
        })

    valid_mask = np.array([x["valid"] for x in candidate_rows], dtype=bool)
    lo, hi = trim_valid_range(valid_mask)
    if lo is None:
        print(f"[WARN] Skip {ep_dir.name}: no valid aligned region")
        return False

    kept_candidates = candidate_rows[lo:hi + 1]

    new_step = 0
    for item in kept_candidates:
        if not item["valid"]:
            continue

        r = item["robot_row"]
        c = item["cmd_row"]
        front_file = item["front_file"]
        wrist_file = item["wrist_file"]
        t_ns = item["t_ns"]

        if copy_images:
            new_front_name = f"{new_step:06d}.png"
            new_wrist_name = f"{new_step:06d}.png"

            if new_front_name not in front_name_map:
                shutil.copy2(front_file, out_front / new_front_name)
                front_name_map[new_front_name] = True
            if new_wrist_name not in wrist_name_map:
                shutil.copy2(wrist_file, out_wrist / new_wrist_name)
                wrist_name_map[new_wrist_name] = True

            front_rel = f"front_rgb/{new_front_name}"
            wrist_rel = f"wrist_rgb/{new_wrist_name}"
        else:
            front_rel = str(front_file.resolve())
            wrist_rel = str(wrist_file.resolve())

        row = {
            "step": new_step,
            "timestamp_ns": int(t_ns),
            "front_rgb": front_rel,
            "wrist_rgb": wrist_rel,

            "robot_timestamp_ns": int(r["timestamp_ns"]),
            "cmd_timestamp_ns": int(c["timestamp_ns"]),
            "front_timestamp_ns": parse_image_timestamp_ns(front_file),
            "wrist_timestamp_ns": parse_image_timestamp_ns(wrist_file),

            "robot_dt_ms": item["robot_diff_ns"] / 1e6,
            "cmd_dt_ms": item["cmd_diff_ns"] / 1e6,
            "front_dt_ms": item["front_diff_ns"] / 1e6,
            "wrist_dt_ms": item["wrist_diff_ns"] / 1e6,

            "j1": float(r["j1"]),
            "j2": float(r["j2"]),
            "j3": float(r["j3"]),
            "j4": float(r["j4"]),
            "j5": float(r["j5"]),
            "j6": float(r["j6"]),
            "gripper_pos": float(r["gripper_pos"]),

            "cmd_tcp_x": float(c["cmd_tcp_x"]),
            "cmd_tcp_y": float(c["cmd_tcp_y"]),
            "cmd_tcp_z": float(c["cmd_tcp_z"]),
            "cmd_tcp_rx": float(c["cmd_tcp_rx"]),
            "cmd_tcp_ry": float(c["cmd_tcp_ry"]),
            "cmd_tcp_rz": float(c["cmd_tcp_rz"]),
            "cmd_gripper": float(c["cmd_gripper"]),
        }
        rows.append(row)
        new_step += 1

    if len(rows) < 5:
        print(f"[WARN] Skip {ep_dir.name}: too few valid output rows")
        return False

    samples_df = pd.DataFrame(rows)
    samples_df.to_csv(out_ep / "samples.csv", index=False)

    meta = {
        "source_episode": ep_dir.name,
        "target_hz": target_hz,
        "num_samples": len(samples_df),
        "time_start_ns": int(samples_df["timestamp_ns"].iloc[0]),
        "time_end_ns": int(samples_df["timestamp_ns"].iloc[-1]),
        "copy_images": copy_images,
        "image_alignment": "nearest_by_real_filename_timestamp",
        "max_state_gap_ms": max_state_gap_ms,
        "max_cmd_gap_ms": max_cmd_gap_ms,
        "max_img_gap_ms": max_img_gap_ms,
        "skip_stats": {
            "skipped_state": skipped_state,
            "skipped_cmd": skipped_cmd,
            "skipped_front": skipped_front,
            "skipped_wrist": skipped_wrist,
        },
    }

    meta_path = ep_dir / "episode_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            raw_meta = json.load(f)
        meta["raw_meta"] = raw_meta

    with open(out_ep / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[OK] {ep_dir.name}: {len(samples_df)} samples | "
        f"state_gap<={max_state_gap_ms}ms cmd_gap<={max_cmd_gap_ms}ms img_gap<={max_img_gap_ms}ms"
    )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, default="./dataset/raw")
    parser.add_argument("--out_root", type=str, default="./dataset/processed")
    parser.add_argument("--target_hz", type=float, default=10.0)
    parser.add_argument("--copy_images", action="store_true")
    parser.add_argument("--max_state_gap_ms", type=float, default=100.0)
    parser.add_argument("--max_cmd_gap_ms", type=float, default=100.0)
    parser.add_argument("--max_img_gap_ms", type=float, default=100.0)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    ep_dirs = get_episode_dirs(raw_root)
    if len(ep_dirs) == 0:
        print("[ERROR] No episodes found.")
        return

    ok = 0
    for ep_dir in ep_dirs:
        success = process_episode(
            ep_dir=ep_dir,
            out_root=out_root,
            target_hz=args.target_hz,
            copy_images=args.copy_images,
            max_state_gap_ms=args.max_state_gap_ms,
            max_cmd_gap_ms=args.max_cmd_gap_ms,
            max_img_gap_ms=args.max_img_gap_ms,
        )
        ok += int(success)

    print(f"\nDone: {ok}/{len(ep_dirs)} episodes processed.")


if __name__ == "__main__":
    main()
