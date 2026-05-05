from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SRot, Slerp


MOTION_KEYS = (
    "body_pos_w",
    "body_quat_w",
    "joint_pos",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "joint_vel",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Convert selected lafan1_gmr2motionlib clips into chair-compatible "
            "motion subdirectories that can be mixed with the existing chair dataset."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=repo_root / "data" / "motion" / "g1" / "lafan1_gmr2motionlib",
        help="Directory containing converted lafan1_gmr2motionlib motion folders.",
    )
    parser.add_argument(
        "--chair-dir",
        type=Path,
        default=repo_root / "data" / "motion" / "g1" / "chair",
        help="Reference chair motion directory providing target meta and static object pose.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where mixed chair motion subdirectories will be written.",
    )
    parser.add_argument(
        "--motion",
        dest="motions",
        action="append",
        required=True,
        help="Motion folder name under --source-root. Repeat this flag for multiple clips.",
    )
    parser.add_argument(
        "--include-chair-ref",
        action="store_true",
        help="Copy the reference chair clip into <output-root>/chair_ref for mixing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing generated motion subdirectories.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)
        f.write("\n")


def lerp(ts_target: np.ndarray, ts_source: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(ts_target, ts_source, x[:, i]) for i in range(x.shape[1])], axis=-1)


def slerp(ts_target: np.ndarray, ts_source: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    batch_shape = quat_wxyz.shape[1:-1]
    quat_dim = quat_wxyz.shape[-1]
    quat_wxyz = quat_wxyz.reshape(quat_wxyz.shape[0], -1, quat_dim)

    out = np.empty((len(ts_target), quat_wxyz.shape[1], quat_dim), dtype=np.float32)
    for i in range(quat_wxyz.shape[1]):
        quat_xyzw = quat_wxyz[:, i, [1, 2, 3, 0]]
        slerp_fn = Slerp(ts_source, SRot.from_quat(quat_xyzw))
        out[:, i, :] = slerp_fn(ts_target).as_quat()[:, [3, 0, 1, 2]]
    return out.reshape(len(ts_target), *batch_shape, quat_dim)


def resample_motion(motion: dict[str, np.ndarray], source_fps: float, target_fps: float) -> dict[str, np.ndarray]:
    if float(source_fps) == float(target_fps):
        return {key: value.astype(np.float32, copy=False) for key, value in motion.items()}

    steps = motion["joint_pos"].shape[0]
    ts_source = np.arange(steps, dtype=np.float64) / float(source_fps)
    target_steps = int(np.floor((steps - 1) * float(target_fps) / float(source_fps))) + 1
    ts_target = np.arange(target_steps, dtype=np.float64) / float(target_fps)

    resampled: dict[str, np.ndarray] = {}
    resampled["body_pos_w"] = lerp(ts_target, ts_source, motion["body_pos_w"].reshape(steps, -1)).reshape(len(ts_target), -1, 3)
    resampled["body_lin_vel_w"] = lerp(ts_target, ts_source, motion["body_lin_vel_w"].reshape(steps, -1)).reshape(len(ts_target), -1, 3)
    resampled["body_quat_w"] = slerp(ts_target, ts_source, motion["body_quat_w"])
    resampled["body_ang_vel_w"] = lerp(ts_target, ts_source, motion["body_ang_vel_w"].reshape(steps, -1)).reshape(len(ts_target), -1, 3)
    resampled["joint_pos"] = lerp(ts_target, ts_source, motion["joint_pos"])
    resampled["joint_vel"] = lerp(ts_target, ts_source, motion["joint_vel"])
    return {key: value.astype(np.float32, copy=False) for key, value in resampled.items()}


def load_motion_dir(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    meta = load_json(path / "meta.json")
    motion = dict(np.load(path / "motion.npz", allow_pickle=True))
    return meta, motion


def build_name_to_index(names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(names)}


def select_source_motion(
    source_meta: dict,
    source_motion: dict[str, np.ndarray],
    target_body_names: list[str],
    target_joint_names: list[str],
) -> dict[str, np.ndarray]:
    source_body_idx = build_name_to_index(source_meta["body_names"])
    source_joint_idx = build_name_to_index(source_meta["joint_names"])

    body_indices = [source_body_idx[name] for name in target_body_names]
    joint_indices = [source_joint_idx[name] for name in target_joint_names]

    return {
        "body_pos_w": source_motion["body_pos_w"][:, body_indices, :],
        "body_quat_w": source_motion["body_quat_w"][:, body_indices, :],
        "joint_pos": source_motion["joint_pos"][:, joint_indices],
        "body_lin_vel_w": source_motion["body_lin_vel_w"][:, body_indices, :],
        "body_ang_vel_w": source_motion["body_ang_vel_w"][:, body_indices, :],
        "joint_vel": source_motion["joint_vel"][:, joint_indices],
    }


def append_static_object(
    motion: dict[str, np.ndarray],
    object_pos: np.ndarray,
    object_quat: np.ndarray,
    object_contact_width: int,
) -> dict[str, np.ndarray]:
    steps = motion["joint_pos"].shape[0]
    object_pos_seq = np.repeat(object_pos[None, None, :], steps, axis=0)
    object_quat_seq = np.repeat(object_quat[None, None, :], steps, axis=0)
    object_lin_vel_seq = np.zeros((steps, 1, 3), dtype=np.float32)
    object_ang_vel_seq = np.zeros((steps, 1, 3), dtype=np.float32)

    mixed = {
        "body_pos_w": np.concatenate([motion["body_pos_w"], object_pos_seq], axis=1),
        "body_quat_w": np.concatenate([motion["body_quat_w"], object_quat_seq], axis=1),
        "joint_pos": motion["joint_pos"],
        "body_lin_vel_w": np.concatenate([motion["body_lin_vel_w"], object_lin_vel_seq], axis=1),
        "body_ang_vel_w": np.concatenate([motion["body_ang_vel_w"], object_ang_vel_seq], axis=1),
        "joint_vel": motion["joint_vel"],
        "object_contact": np.zeros((steps, object_contact_width), dtype=bool),
    }
    return mixed


def prepare_output_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} already exists. Pass --force to overwrite it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_reference_clip(chair_dir: Path, output_root: Path, force: bool) -> None:
    dst_dir = output_root / "chair_ref"
    prepare_output_dir(dst_dir, force=force)
    for name in ("motion.npz", "meta.json", "annotation_summary.json"):
        src = chair_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


def main() -> None:
    args = parse_args()

    chair_meta, chair_motion = load_motion_dir(args.chair_dir)
    chair_summary_path = args.chair_dir / "annotation_summary.json"
    chair_summary = load_json(chair_summary_path) if chair_summary_path.exists() else {}

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    object_body_name = chair_meta["body_names"][-1]
    robot_body_names = chair_meta["body_names"][:-1]
    target_joint_names = chair_meta["joint_names"]
    target_fps = float(chair_meta["fps"])

    chair_object_idx = len(chair_meta["body_names"]) - 1
    object_pos = chair_motion["body_pos_w"][0, chair_object_idx].astype(np.float32, copy=False)
    object_quat = chair_motion["body_quat_w"][0, chair_object_idx].astype(np.float32, copy=False)
    object_contact_width = int(chair_motion["object_contact"].shape[1]) if "object_contact" in chair_motion else 1

    generated = []

    if args.include_chair_ref:
        copy_reference_clip(args.chair_dir, output_root, force=args.force)
        generated.append("chair_ref")

    for motion_name in args.motions:
        source_dir = args.source_root / motion_name
        if not source_dir.exists():
            raise FileNotFoundError(f"Source motion directory does not exist: {source_dir}")

        source_meta, source_motion = load_motion_dir(source_dir)
        trimmed_motion = select_source_motion(source_meta, source_motion, robot_body_names, target_joint_names)
        resampled_motion = resample_motion(trimmed_motion, float(source_meta["fps"]), target_fps)
        mixed_motion = append_static_object(
            resampled_motion,
            object_pos=object_pos,
            object_quat=object_quat,
            object_contact_width=object_contact_width,
        )

        dst_dir = output_root / motion_name
        prepare_output_dir(dst_dir, force=args.force)
        np.savez_compressed(dst_dir / "motion.npz", **mixed_motion)
        save_json(dst_dir / "meta.json", chair_meta)

        summary = {
            "num_frames": int(mixed_motion["joint_pos"].shape[0]),
            "num_positive_frames": 0,
            "positive_first_frame": -1,
            "positive_last_frame": -1,
            "start_frame_index": 0,
            "object_scale": chair_summary.get("object_scale", 1.0),
            "is_static_object": True,
            "reference_object_body_name": object_body_name,
            "source_motion_name": motion_name,
            "source_fps": float(source_meta["fps"]),
            "target_fps": target_fps,
        }
        save_json(dst_dir / "annotation_summary.json", summary)
        generated.append(motion_name)

    manifest = {
        "source_root": str(args.source_root),
        "chair_dir": str(args.chair_dir),
        "output_root": str(output_root),
        "motions": generated,
        "target_fps": target_fps,
        "meta_body_names": chair_meta["body_names"],
        "meta_joint_names": chair_meta["joint_names"],
    }
    save_json(output_root / "mix_manifest.json", manifest)

    print(f"Wrote {len(generated)} motion directories under {output_root}")


if __name__ == "__main__":
    main()
