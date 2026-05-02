from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_BODY_NAMES = [
    "pelvis",
    "head_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_rubber_hand",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_rubber_hand",
]

DEFAULT_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

REQUIRED_KEYS = {
    "global_translation",
    "global_rotation",
    "global_velocity",
    "global_angular_velocity",
    "dof_pos",
    "dof_vels",
    "fps",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Convert GMR2motionlib torch-serialized motion files into HDMI "
            "motion directories containing motion.npz + meta.json."
        )
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path(r"D:\VSCODE"),
        help="Directory containing the source *.npy motion files.",
    )
    parser.add_argument(
        "--dst-dir",
        type=Path,
        default=repo_root / "data" / "motion" / "g1" / "lafan1_gmr2motionlib",
        help="Directory where converted HDMI motion folders will be written.",
    )
    parser.add_argument(
        "--glob",
        default="*.npy",
        help="Glob used to find source motion files inside --src-dir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of source files to convert.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing motion.npz and meta.json files if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the conversion plan without writing files.",
    )
    parser.add_argument(
        "--meta-template",
        type=Path,
        default=None,
        help=(
            "Optional JSON file containing body_names and joint_names. "
            "If omitted, the built-in inferred G1 template is used."
        ),
    )
    return parser.parse_args()


def load_meta_template(path: Path | None) -> tuple[list[str], list[str]]:
    if path is None:
        return list(DEFAULT_BODY_NAMES), list(DEFAULT_JOINT_NAMES)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        body_names = list(data["body_names"])
        joint_names = list(data["joint_names"])
    except KeyError as exc:
        raise KeyError(f"{path} must contain body_names and joint_names") from exc

    return body_names, joint_names


def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(value)!r}")


def validate_source_motion(source_path: Path, motion: dict[str, Any], body_names: list[str], joint_names: list[str]) -> None:
    missing = sorted(REQUIRED_KEYS - set(motion.keys()))
    if missing:
        raise KeyError(f"{source_path} is missing required keys: {missing}")

    body_count = tensor_to_numpy(motion["global_translation"]).shape[1]
    joint_count = tensor_to_numpy(motion["dof_pos"]).shape[1]
    if body_count != len(body_names):
        raise ValueError(
            f"{source_path} body dimension {body_count} does not match template body_names "
            f"length {len(body_names)}"
        )
    if joint_count != len(joint_names):
        raise ValueError(
            f"{source_path} joint dimension {joint_count} does not match template joint_names "
            f"length {len(joint_names)}"
        )


def convert_motion(motion: dict[str, Any]) -> dict[str, np.ndarray]:
    converted = {
        "body_pos_w": tensor_to_numpy(motion["global_translation"]).astype(np.float32, copy=False),
        "body_quat_w": tensor_to_numpy(motion["global_rotation"]).astype(np.float32, copy=False),
        "joint_pos": tensor_to_numpy(motion["dof_pos"]).astype(np.float32, copy=False),
        "body_lin_vel_w": tensor_to_numpy(motion["global_velocity"]).astype(np.float32, copy=False),
        "body_ang_vel_w": tensor_to_numpy(motion["global_angular_velocity"]).astype(np.float32, copy=False),
        "joint_vel": tensor_to_numpy(motion["dof_vels"]).astype(np.float32, copy=False),
    }
    return converted


def write_motion_dir(
    source_path: Path,
    output_root: Path,
    body_names: list[str],
    joint_names: list[str],
    overwrite: bool,
    dry_run: bool,
) -> tuple[Path, int]:
    motion_name = source_path.stem
    motion_dir = output_root / motion_name
    motion_path = motion_dir / "motion.npz"
    meta_path = motion_dir / "meta.json"

    if not overwrite and motion_path.exists() and meta_path.exists():
        return motion_dir, 0

    source_motion = torch.load(source_path, map_location="cpu")
    if not isinstance(source_motion, dict):
        raise TypeError(f"{source_path} did not deserialize to a dict")

    validate_source_motion(source_path, source_motion, body_names, joint_names)
    converted = convert_motion(source_motion)

    fps_value = float(source_motion["fps"])
    meta = {
        "body_names": body_names,
        "joint_names": joint_names,
        "fps": fps_value,
    }

    if dry_run:
        return motion_dir, int(converted["body_pos_w"].shape[0])

    motion_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(motion_path, **converted)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True)

    return motion_dir, int(converted["body_pos_w"].shape[0])


def main() -> None:
    args = parse_args()
    body_names, joint_names = load_meta_template(args.meta_template)

    if not args.src_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {args.src_dir}")

    source_paths = sorted(args.src_dir.glob(args.glob))
    if args.limit is not None:
        source_paths = source_paths[: args.limit]

    if not source_paths:
        raise RuntimeError(f"No source files matched {args.glob!r} under {args.src_dir}")

    args.dst_dir.mkdir(parents=True, exist_ok=True)

    converted_count = 0
    skipped_count = 0
    total_frames = 0
    converted_dirs: list[str] = []

    for source_path in source_paths:
        motion_dir, frame_count = write_motion_dir(
            source_path=source_path,
            output_root=args.dst_dir,
            body_names=body_names,
            joint_names=joint_names,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        if frame_count == 0:
            skipped_count += 1
            continue
        converted_count += 1
        total_frames += frame_count
        converted_dirs.append(str(motion_dir))
        print(f"[convert] {source_path.name} -> {motion_dir}")

    manifest = {
        "src_dir": str(args.src_dir),
        "dst_dir": str(args.dst_dir),
        "glob": args.glob,
        "body_count": len(body_names),
        "joint_count": len(joint_names),
        "converted_count": converted_count,
        "skipped_count": skipped_count,
        "total_frames": total_frames,
        "dry_run": args.dry_run,
        "motions": converted_dirs,
    }

    manifest_path = args.dst_dir / "conversion_manifest.json"
    if not args.dry_run:
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=True)

    print(
        "[summary] "
        f"converted={converted_count} skipped={skipped_count} "
        f"total_frames={total_frames} dst={args.dst_dir}"
    )
    if not args.dry_run:
        print(f"[summary] manifest={manifest_path}")


if __name__ == "__main__":
    main()
