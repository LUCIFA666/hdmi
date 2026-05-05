from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    alias: str
    path: str
    description: str
    recommended: str


DATASETS: dict[str, DatasetSpec] = {
    "chair": DatasetSpec(
        alias="chair",
        path="data/motion/g1/chair",
        description="Original single chair sit-down clip",
        recommended="Task-specific ref finetuning",
    ),
    "walk": DatasetSpec(
        alias="walk",
        path="data/motion/g1/chair_mix_walk/.*",
        description="chair_ref plus 3 walk clips",
        recommended="Tracking warmup or lightweight smoke test",
    ),
    "locomotion": DatasetSpec(
        alias="locomotion",
        path="data/motion/g1/chair_mix_locomotion/.*",
        description="chair_ref plus walk/run/sprint clips",
        recommended="Best default for no-chair teacher warmup",
    ),
    "obstacles": DatasetSpec(
        alias="obstacles",
        path="data/motion/g1/chair_mix_obstacles/.*",
        description="chair_ref plus jumps/obstacles/multipleActions clips",
        recommended="Robustness boost or second-stage augmentation",
    ),
}


TASKS = {
    "no_chair": "G1/hdmi/custom_object_011_no_chair",
    "ref": "G1/hdmi/custom_object_011_ref",
}


TRAIN_SCRIPT = "scripts/train.py"
PLAY_SCRIPT = "scripts/play.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch replay, play, and training for custom_object_011 datasets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List built-in dataset aliases.")
    subparsers.add_parser("recipe", help="Show the recommended training recipe.")

    for name, help_text in (
        ("replay-no-chair", "Replay reference motion with the no-chair task."),
        ("replay-ref", "Replay reference motion with the chair object spawned."),
        ("train-no-chair", "Train the pure tracking policy on a chosen dataset."),
        ("train-ref", "Train or finetune the chair-aware tracking policy."),
        ("play-no-chair", "Play a trained no-chair checkpoint."),
        ("play-ref", "Play a trained ref checkpoint."),
    ):
        cmd_parser = subparsers.add_parser(name, help=help_text)
        cmd_parser.add_argument("dataset", choices=sorted(DATASETS.keys()))
        cmd_parser.add_argument(
            "--checkpoint",
            type=str,
            default=None,
            help="Checkpoint path or run:<entity/project/run_id> when required.",
        )
        cmd_parser.add_argument(
            "--algo",
            type=str,
            default=None,
            help="Hydra algo override. Defaults depend on the command.",
        )
        cmd_parser.add_argument(
            "--python-exe",
            type=str,
            default=os.environ.get("PYTHON_EXE") or sys.executable or "python",
            help="Python executable used to launch the underlying script.",
        )
        cmd_parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print the resolved command without executing it.",
        )

    return parser


def dataset_spec(alias: str) -> DatasetSpec:
    return DATASETS[alias]


def sanitize_overrides(overrides: list[str]) -> list[str]:
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]
    return overrides


def default_algo(command: str, checkpoint: str | None) -> str:
    if command == "train-ref":
        return "ppo_roa_finetune" if checkpoint else "ppo_roa_train"
    if command == "play-ref":
        return "ppo_roa_finetune"
    return "ppo_roa_train"


def default_exp_name(command: str, dataset_alias: str) -> str:
    if command == "train-no-chair":
        return f"custom_object_011_no_chair_{dataset_alias}"
    if command == "train-ref":
        return f"custom_object_011_ref_{dataset_alias}"
    return ""


def build_base_command(
    python_exe: str,
    script: str,
    algo: str,
    task: str,
    data_path: str,
) -> list[str]:
    cmd = [
        python_exe,
        script,
        f"algo={algo}",
        f"task={task}",
        f"task.command.data_path={data_path}",
    ]
    wandb_mode = os.environ.get("WANDB_MODE")
    if wandb_mode:
        cmd.append(f"wandb.mode={wandb_mode}")
    return cmd


def build_command(args: argparse.Namespace, passthrough_overrides: list[str]) -> list[str]:
    spec = dataset_spec(args.dataset)
    overrides = sanitize_overrides(passthrough_overrides)
    algo = args.algo or default_algo(args.command, args.checkpoint)

    if args.command == "replay-no-chair":
        cmd = build_base_command(args.python_exe, PLAY_SCRIPT, algo, TASKS["no_chair"], spec.path)
        cmd.extend(["+task.command.replay_motion=true", "task.num_envs=1"])
    elif args.command == "replay-ref":
        cmd = build_base_command(args.python_exe, PLAY_SCRIPT, algo, TASKS["ref"], spec.path)
        cmd.extend(["+task.command.replay_motion=true", "task.num_envs=1"])
    elif args.command == "train-no-chair":
        cmd = build_base_command(args.python_exe, TRAIN_SCRIPT, algo, TASKS["no_chair"], spec.path)
        exp_name = default_exp_name(args.command, args.dataset)
        if exp_name:
            cmd.append(f"exp_name={exp_name}")
        if args.checkpoint:
            cmd.append(f"checkpoint_path={args.checkpoint}")
    elif args.command == "train-ref":
        cmd = build_base_command(args.python_exe, TRAIN_SCRIPT, algo, TASKS["ref"], spec.path)
        exp_name = default_exp_name(args.command, args.dataset)
        if exp_name:
            cmd.append(f"exp_name={exp_name}")
        if args.checkpoint:
            cmd.append(f"checkpoint_path={args.checkpoint}")
    elif args.command == "play-no-chair":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for play-no-chair")
        cmd = build_base_command(args.python_exe, PLAY_SCRIPT, algo, TASKS["no_chair"], spec.path)
        cmd.extend(["task.num_envs=1", f"checkpoint_path={args.checkpoint}"])
    elif args.command == "play-ref":
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for play-ref")
        cmd = build_base_command(args.python_exe, PLAY_SCRIPT, algo, TASKS["ref"], spec.path)
        cmd.extend(["task.num_envs=1", f"checkpoint_path={args.checkpoint}"])
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    cmd.extend(overrides)
    return cmd


def print_command(cmd: list[str]) -> None:
    print("Resolved command:")
    print("  " + subprocess.list2cmdline(cmd))


def command_examples() -> list[str]:
    return [
        "custom_object_011_pipeline.cmd replay-no-chair locomotion",
        "custom_object_011_pipeline.cmd train-no-chair locomotion -- wandb.mode=online total_frames=50000000",
        "custom_object_011_pipeline.cmd train-ref chair --checkpoint run:<teacher_run_path> --algo ppo_roa_finetune",
        "custom_object_011_pipeline.cmd play-ref chair --checkpoint run:<student_run_path> --algo ppo_roa_finetune",
    ]


def print_datasets() -> None:
    print("Available dataset aliases:")
    for alias, spec in DATASETS.items():
        print(f"- {alias}: {spec.path}")
        print(f"  {spec.description}")
        print(f"  Recommended use: {spec.recommended}")


def print_recipe() -> None:
    print("Recommended custom_object_011 training recipe:")
    print("1. Smoke test the locomotion mix with reference replay.")
    print("   custom_object_011_pipeline.cmd replay-no-chair locomotion")
    print("2. Warm up a teacher on the no-chair task with the locomotion mix.")
    print("   custom_object_011_pipeline.cmd train-no-chair locomotion -- wandb.mode=online")
    print("3. Optionally add robustness with the obstacles mix.")
    print("   custom_object_011_pipeline.cmd train-no-chair obstacles --checkpoint run:<teacher_run_path> --algo ppo_roa_finetune")
    print("4. Finetune the chair-aware policy on the original chair clip.")
    print("   custom_object_011_pipeline.cmd train-ref chair --checkpoint run:<teacher_run_path> --algo ppo_roa_finetune")
    print("5. Visualize the resulting policy.")
    print("   custom_object_011_pipeline.cmd play-ref chair --checkpoint run:<student_run_path> --algo ppo_roa_finetune")


def main() -> int:
    parser = build_parser()
    args, passthrough_overrides = parser.parse_known_args()

    if args.command == "list":
        print_datasets()
        print("\nExamples:")
        for example in command_examples():
            print(f"- {example}")
        return 0

    if args.command == "recipe":
        print_recipe()
        return 0

    cmd = build_command(args, passthrough_overrides)
    spec = dataset_spec(args.dataset)
    print(f"[pipeline] Dataset alias: {spec.alias}")
    print(f"[pipeline] Dataset path : {spec.path}")
    print(f"[pipeline] Description  : {spec.description}")
    print_command(cmd)

    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
