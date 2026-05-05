from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.train_sequential import run_training_stage


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "cfg"

DATASET_PATHS = {
    "chair": "data/motion/g1/chair",
    "walk": "data/motion/g1/chair_mix_walk/.*",
    "locomotion": "data/motion/g1/chair_mix_locomotion/.*",
    "obstacles": "data/motion/g1/chair_mix_obstacles/.*",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the recommended custom_object_011 curriculum: "
            "train-no-chair on a mixed dataset, then finetune train-ref on chair."
        )
    )
    parser.add_argument(
        "--teacher-dataset",
        choices=sorted(DATASET_PATHS.keys()),
        default="locomotion",
        help="Dataset alias used for the stage-1 no-chair teacher.",
    )
    parser.add_argument(
        "--ref-dataset",
        choices=sorted(DATASET_PATHS.keys()),
        default="chair",
        help="Dataset alias used for the stage-2 ref finetune.",
    )
    parser.add_argument(
        "--teacher-algo",
        default="ppo_roa_train",
        help="Algo config used for the stage-1 teacher.",
    )
    parser.add_argument(
        "--ref-algo",
        default="ppo_roa_finetune",
        help="Algo config used for the stage-2 finetune.",
    )
    parser.add_argument(
        "--teacher-total-frames",
        type=int,
        default=None,
        help="Optional total_frames override for the stage-1 teacher.",
    )
    parser.add_argument(
        "--ref-total-frames",
        type=int,
        default=None,
        help="Optional total_frames override for the stage-2 ref finetune.",
    )
    parser.add_argument(
        "--teacher-override",
        action="append",
        default=[],
        help="Extra Hydra override applied only to the stage-1 teacher. Repeatable.",
    )
    parser.add_argument(
        "--ref-override",
        action="append",
        default=[],
        help="Extra Hydra override applied only to the stage-2 ref finetune. Repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print both stage overrides and exit without starting training.",
    )
    return parser


def compose_stage_cfg(overrides: list[str]):
    with initialize_config_dir(config_dir=str(CONFIG_PATH), version_base=None):
        cfg = compose(config_name="train", overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def launch_stage(stage_label: str, overrides: list[str]) -> str:
    print("=" * 80)
    print(f"Launching {stage_label}")
    for override in overrides:
        print(f"  {override}")
    print("=" * 80)

    cfg = compose_stage_cfg(overrides)
    return_queue = multiprocessing.Queue(1)
    process = multiprocessing.Process(
        target=run_training_stage,
        kwargs={"cfg": cfg, "return_queue": return_queue},
    )
    process.start()
    process.join()

    if process.exitcode != 0:
        raise RuntimeError(f"{stage_label} failed with exit code {process.exitcode}")

    try:
        return return_queue.get(timeout=10)
    except Exception as exc:  # pragma: no cover - defensive path
        raise RuntimeError(f"{stage_label} completed but did not return a run path") from exc


def main() -> int:
    parser = build_parser()
    args, shared_overrides = parser.parse_known_args()

    teacher_dataset_path = DATASET_PATHS[args.teacher_dataset]
    ref_dataset_path = DATASET_PATHS[args.ref_dataset]

    teacher_overrides = list(shared_overrides)
    teacher_overrides.extend(
        [
            f"algo={args.teacher_algo}",
            "task=G1/hdmi/custom_object_011_no_chair",
            f"task.command.data_path={teacher_dataset_path}",
            f"exp_name=custom_object_011_no_chair_{args.teacher_dataset}",
        ]
    )
    if args.teacher_total_frames is not None:
        teacher_overrides.append(f"total_frames={args.teacher_total_frames}")
    teacher_overrides.extend(args.teacher_override)

    if args.dry_run:
        print("Stage 1 teacher overrides:")
        for override in teacher_overrides:
            print(f"  {override}")
        print("\nStage 2 overrides will be printed after the teacher run path is available.")
        return 0

    teacher_run_path = launch_stage("stage 1 / teacher", teacher_overrides)
    print(f"[curriculum] Teacher run path: {teacher_run_path}")

    ref_overrides = list(shared_overrides)
    ref_overrides.extend(
        [
            f"algo={args.ref_algo}",
            "task=G1/hdmi/custom_object_011_ref",
            f"task.command.data_path={ref_dataset_path}",
            f"exp_name=custom_object_011_ref_{args.ref_dataset}",
            f"checkpoint_path=run:{teacher_run_path}",
        ]
    )
    if args.ref_total_frames is not None:
        ref_overrides.append(f"total_frames={args.ref_total_frames}")
    ref_overrides.extend(args.ref_override)

    ref_run_path = launch_stage("stage 2 / ref finetune", ref_overrides)
    print(f"[curriculum] Ref run path: {ref_run_path}")
    print("[curriculum] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
