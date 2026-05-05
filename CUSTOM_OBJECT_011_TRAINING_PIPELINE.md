# custom_object_011 Training Pipeline

这份说明对应下面这个统一入口：

- [custom_object_011_pipeline.cmd](C:/Users/17355/Desktop/VSCODE/hdmi/custom_object_011_pipeline.cmd)

它把 `chair / chair_mix_walk / chair_mix_locomotion / chair_mix_obstacles` 这几份数据集的常用 `replay / train / play` 命令都收口到了一个地方，适合直接上手。

## 1. 先看可用数据集

```bat
custom_object_011_pipeline.cmd list
```

内置别名：

- `chair` -> `data/motion/g1/chair`
- `walk` -> `data/motion/g1/chair_mix_walk/.*`
- `locomotion` -> `data/motion/g1/chair_mix_locomotion/.*`
- `obstacles` -> `data/motion/g1/chair_mix_obstacles/.*`

其中：

- `locomotion` 最适合做 `no_chair` teacher 预热
- `obstacles` 适合在 teacher 基础上补鲁棒性
- `chair` 适合最后做 `ref` 任务专项微调

## 2. 最推荐的训练顺序

### 第一步：先回放 reference，确认数据读得对

```bat
custom_object_011_pipeline.cmd replay-no-chair locomotion
custom_object_011_pipeline.cmd replay-ref chair
```

### 第二步：训练 no_chair teacher

```bat
custom_object_011_pipeline.cmd train-no-chair locomotion -- wandb.mode=online
```

如果你先想本地试跑，可以先把 W&B 关掉，并缩短总帧数：

```bat
custom_object_011_pipeline.cmd train-no-chair locomotion -- wandb.mode=disabled total_frames=20000000
```

### 第三步：可选地再用 obstacles 混合数据补鲁棒性

把上一步得到的 teacher run path 填进去：

```bat
custom_object_011_pipeline.cmd train-no-chair obstacles --checkpoint run:<teacher_run_path> --algo ppo_roa_finetune
```

### 第四步：切到 chair 做 ref 微调

```bat
custom_object_011_pipeline.cmd train-ref chair --checkpoint run:<teacher_run_path> --algo ppo_roa_finetune
```

如果你希望直接从头训练 `ref`，也可以不带 `--checkpoint`：

```bat
custom_object_011_pipeline.cmd train-ref chair -- wandb.mode=online
```

## 3. 训练后播放 checkpoint

### 播放 no_chair

```bat
custom_object_011_pipeline.cmd play-no-chair locomotion --checkpoint run:<teacher_run_path>
```

### 播放 ref

```bat
custom_object_011_pipeline.cmd play-ref chair --checkpoint run:<student_run_path> --algo ppo_roa_finetune
```

## 4. 常用调参入口

所有额外 Hydra override 都可以放在命令最后，并建议用 `--` 和前面的脚本参数隔开。

例如：

```bat
custom_object_011_pipeline.cmd train-no-chair locomotion -- ^
  wandb.mode=disabled ^
  total_frames=50000000 ^
  task.num_envs=1024 ^
  save_interval=100
```

常见可调项：

- `wandb.mode=disabled`
- `total_frames=50000000`
- `task.num_envs=1024`
- `headless=false`
- `save_interval=100`
- `seed=1`

## 5. 两个实用建议

- 显存先不确定时，优先把 `task.num_envs` 从默认值往下调，比如 `2048 -> 1024 -> 512`
- 真正任务效果主要看最后的 `train-ref chair`，前面的 `locomotion / obstacles` 更像是在帮策略把 tracking 基础打稳
