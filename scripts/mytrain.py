from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def _parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Custom motion directory containing motion.npz and meta.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the rewritten Hydra arguments and exit without training.",
    )
    return parser.parse_known_args(argv)


def _normalize_motion_dir(raw_path: str) -> str:
    motion_dir = Path(raw_path).expanduser()
    if not motion_dir.is_absolute():
        cwd_candidate = (Path.cwd() / motion_dir).resolve()
        repo_candidate = (REPO_ROOT / motion_dir).resolve()
        if cwd_candidate.exists():
            motion_dir = cwd_candidate
        else:
            motion_dir = repo_candidate
    else:
        motion_dir = motion_dir.resolve()

    if motion_dir.is_file():
        if motion_dir.name not in {"motion.npz", "meta.json"}:
            raise FileNotFoundError(
                f"Expected a motion directory or one of motion.npz/meta.json, got: {motion_dir}"
            )
        motion_dir = motion_dir.parent

    motion_path = motion_dir / "motion.npz"
    meta_path = motion_dir / "meta.json"
    missing_files = [path.name for path in (motion_path, meta_path) if not path.exists()]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(f"{motion_dir} is missing required files: {missing}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    required_keys = ["body_names", "joint_names", "fps"]
    missing_keys = [key for key in required_keys if key not in meta]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise KeyError(f"{meta_path} is missing required keys: {missing}")

    try:
        return motion_dir.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return motion_dir.as_posix()


def _strip_existing_data_path_override(args: list[str]) -> list[str]:
    prefixes = ("task.command.data_path=", "+task.command.data_path=")
    return [arg for arg in args if not arg.startswith(prefixes)]


def main() -> None:
    wrapper_args, hydra_args = _parse_wrapper_args(sys.argv[1:])

    if wrapper_args.data_dir:
        custom_data_path = _normalize_motion_dir(wrapper_args.data_dir)
        hydra_args = _strip_existing_data_path_override(hydra_args)
        hydra_args.append(f"task.command.data_path={custom_data_path}")
        print(f"[mytrain] Using custom dataset: {custom_data_path}")

    if wrapper_args.dry_run:
        print("[mytrain] Final Hydra args:")
        for arg in hydra_args:
            print(f"  {arg}")
        return

    sys.argv = [sys.argv[0], *hydra_args]

    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
