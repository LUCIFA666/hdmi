from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new motion directory whose world-frame poses are rotated so "
            "the reference object is upright while preserving its yaw."
        )
    )
    parser.add_argument("src_dir", type=Path, help="Source motion directory containing motion.npz and meta.json.")
    parser.add_argument("dst_dir", type=Path, help="Destination directory to create.")
    parser.add_argument(
        "--object-body-name",
        type=str,
        default=None,
        help="Object body name in meta.json. Defaults to the last body in body_names.",
    )
    return parser.parse_args()


def _wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    return quat[..., [1, 2, 3, 0]]


def _xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return quat[..., [3, 0, 1, 2]]


def _rotation_to_yaw_only(rotation: R) -> R:
    matrix = rotation.as_matrix()
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    return R.from_euler("z", yaw)


def _load_meta(meta_path: Path) -> dict:
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_object_body_index(meta: dict, object_body_name: str | None) -> int:
    body_names = meta["body_names"]
    if object_body_name is None:
        return len(body_names) - 1
    try:
        return body_names.index(object_body_name)
    except ValueError as exc:
        raise ValueError(f"Object body '{object_body_name}' not found in meta.json.") from exc


def _transform_motion(motion: dict[str, np.ndarray], object_body_index: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    body_pos_w = motion["body_pos_w"]
    body_quat_w = motion["body_quat_w"]
    body_lin_vel_w = motion["body_lin_vel_w"]
    body_ang_vel_w = motion["body_ang_vel_w"]

    object_quat0 = body_quat_w[0, object_body_index]
    object_pos0 = body_pos_w[0, object_body_index]

    object_rot0 = R.from_quat(_wxyz_to_xyzw(object_quat0))
    yaw_rot = _rotation_to_yaw_only(object_rot0)
    correction = yaw_rot * object_rot0.inv()

    pos_shape = body_pos_w.shape
    quat_shape = body_quat_w.shape
    vel_shape = body_lin_vel_w.shape
    ang_vel_shape = body_ang_vel_w.shape

    centered_positions = (body_pos_w - object_pos0).reshape(-1, 3)
    rotated_positions = correction.apply(centered_positions).reshape(pos_shape) + object_pos0

    rotations = R.from_quat(_wxyz_to_xyzw(body_quat_w.reshape(-1, 4)))
    rotated_quats = _xyzw_to_wxyz((correction * rotations).as_quat()).reshape(quat_shape)

    rotated_lin_vel = correction.apply(body_lin_vel_w.reshape(-1, 3)).reshape(vel_shape)
    rotated_ang_vel = correction.apply(body_ang_vel_w.reshape(-1, 3)).reshape(ang_vel_shape)

    transformed = dict(motion)
    transformed["body_pos_w"] = rotated_positions.astype(body_pos_w.dtype, copy=False)
    transformed["body_quat_w"] = rotated_quats.astype(body_quat_w.dtype, copy=False)
    transformed["body_lin_vel_w"] = rotated_lin_vel.astype(body_lin_vel_w.dtype, copy=False)
    transformed["body_ang_vel_w"] = rotated_ang_vel.astype(body_ang_vel_w.dtype, copy=False)

    object_rot_new = R.from_quat(_wxyz_to_xyzw(transformed["body_quat_w"][0, object_body_index]))
    object_euler_new = object_rot_new.as_euler("xyz", degrees=True)
    object_yaw_old = _rotation_to_yaw_only(object_rot0).as_euler("xyz", degrees=True)[2]
    object_yaw_new = _rotation_to_yaw_only(object_rot_new).as_euler("xyz", degrees=True)[2]

    summary = {
        "object_roll_deg": float(object_euler_new[0]),
        "object_pitch_deg": float(object_euler_new[1]),
        "object_yaw_old_deg": float(object_yaw_old),
        "object_yaw_new_deg": float(object_yaw_new),
        "object_pos_drift": float(
            np.linalg.norm(transformed["body_pos_w"][:, object_body_index] - body_pos_w[:, object_body_index], axis=1).max()
        ),
        "max_relative_distance_error_3d": float(
            np.abs(
                np.linalg.norm(body_pos_w[:, 0] - body_pos_w[:, object_body_index], axis=1)
                - np.linalg.norm(
                    transformed["body_pos_w"][:, 0] - transformed["body_pos_w"][:, object_body_index],
                    axis=1,
                )
            ).max()
        ),
        "max_relative_distance_error_xy": float(
            np.abs(
                np.linalg.norm(body_pos_w[:, 0, :2] - body_pos_w[:, object_body_index, :2], axis=1)
                - np.linalg.norm(
                    transformed["body_pos_w"][:, 0, :2] - transformed["body_pos_w"][:, object_body_index, :2],
                    axis=1,
                )
            ).max()
        ),
    }
    return transformed, summary


def _copy_support_files(src_dir: Path, dst_dir: Path) -> None:
    for name in ("meta.json", "annotation_summary.json"):
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


def main() -> None:
    args = _parse_args()
    src_dir = args.src_dir.resolve()
    dst_dir = args.dst_dir.resolve()

    motion_path = src_dir / "motion.npz"
    meta_path = src_dir / "meta.json"
    if not motion_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"{src_dir} must contain motion.npz and meta.json")

    meta = _load_meta(meta_path)
    object_body_index = _resolve_object_body_index(meta, args.object_body_name)

    with np.load(motion_path) as motion_file:
        motion = {key: motion_file[key] for key in motion_file.files}

    transformed, summary = _transform_motion(motion, object_body_index)

    dst_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst_dir / "motion.npz", **transformed)
    _copy_support_files(src_dir, dst_dir)

    body_name = meta["body_names"][object_body_index]
    print(f"[upright_motion_by_object] Wrote {dst_dir / 'motion.npz'}")
    print(f"[upright_motion_by_object] Object body: {body_name}")
    print(
        "[upright_motion_by_object] Object euler after correction "
        f"(deg xyz): roll={summary['object_roll_deg']:.4f}, pitch={summary['object_pitch_deg']:.4f}, "
        f"yaw(old/new)={summary['object_yaw_old_deg']:.4f}/{summary['object_yaw_new_deg']:.4f}"
    )
    print(f"[upright_motion_by_object] Object max position drift: {summary['object_pos_drift']:.8f}")
    print(
        "[upright_motion_by_object] Max robot-object 3D distance error: "
        f"{summary['max_relative_distance_error_3d']:.8f}"
    )
    print(
        "[upright_motion_by_object] Max robot-object XY projection difference: "
        f"{summary['max_relative_distance_error_xy']:.8f}"
    )


if __name__ == "__main__":
    main()
