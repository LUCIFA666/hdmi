# HDMI 自定义数据接入流程

这份文档总结了把“自己的物体 + 自己的 motion 数据”接到当前 `hdmi1` 仓库里，并完成回放、渲染、训练的完整流程。本文档以当前已经接通的 `chair / chair1 / custom_object_011_ref` 为例。

## 1. 目标

目标是让 HDMI 使用我们自己的对象资产和动作数据，并能完成下面几件事：

- 在 Isaac Sim 中 `play` reference motion
- 在 Isaac Sim 中 `render` 视频
- 用自定义 motion 数据进行训练
- 在发现数据整体歪斜时，按对象姿态生成一份“扶正后”的新数据目录

---

## 2. 当前示例里用到的关键文件

### 2.1 对象资产

- 资产注册文件：
  [active_adaptation/assets/objects.py](C:/Users/17355/Desktop/VSCODE/hdmi1/active_adaptation/assets/objects.py:303)

当前 `chair` 的注册方式是：

```python
CHAIR_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/chair",
    spawn=UsdFileCfg(
        usd_path=f"{ASSET_PATH}/objects/chair/object.usda",
```

也就是说：

- 场景里的对象名字是 `chair`
- 对应 USD 入口文件是 `active_adaptation/assets/objects/chair/object.usda`

### 2.2 动作数据目录

原始数据目录：

- [data/motion/g1/chair](C:/Users/17355/Desktop/VSCODE/hdmi1/data/motion/g1/chair)

扶正后的新目录：

- [data/motion/g1/chair1](C:/Users/17355/Desktop/VSCODE/hdmi1/data/motion/g1/chair1)

每个 motion 目录至少需要：

- `motion.npz`
- `meta.json`
- `annotation_summary.json`（推荐保留）

### 2.3 自定义任务配置

本次示例使用的任务配置：

- [cfg/task/G1/hdmi/custom_object_011_ref.yaml](C:/Users/17355/Desktop/VSCODE/hdmi1/cfg/task/G1/hdmi/custom_object_011_ref.yaml:1)

关键配置如下：

```yaml
command:
  data_path: data/motion/g1/chair
  root_body_name: "pelvis"
  object_asset_name: "chair"
  object_motion_name: "custom_object_011"
  object_body_name: "object"
```

含义：

- `data_path`：指向 motion 数据目录
- `object_asset_name`：场景中加载哪个对象资产
- `object_motion_name`：motion 数据里对象 body 的名字
- `object_body_name`：USD / 仿真里对象 body 的名字

---

## 3. 数据格式要求

HDMI 会读取 `motion.npz + meta.json`。当前仓库要求的数据结构与 [README.md](C:/Users/17355/Desktop/VSCODE/hdmi1/README.md:53) 一致。

### 3.1 `meta.json`

至少需要：

- `body_names`
- `joint_names`
- `fps`

本次 `chair` 的 `meta.json` 中：

- root body 是 `pelvis`
- 最后一个 body 是对象 `custom_object_011`

### 3.2 `motion.npz`

至少包含这些 key：

- `body_pos_w`
- `body_quat_w`
- `joint_pos`
- `body_lin_vel_w`
- `body_ang_vel_w`
- `joint_vel`
- `object_contact`（如果已有，建议保留）

形状一般为：

- `body_pos_w`: `[T, B, 3]`
- `body_quat_w`: `[T, B, 4]`
- `joint_pos`: `[T, J]`

其中 `body_quat_w` 在这套代码里按 `wxyz` 处理。

---

## 4. 把自己的对象资产接进来

如果你有自己的对象 USD，需要先在 [active_adaptation/assets/objects.py](C:/Users/17355/Desktop/VSCODE/hdmi1/active_adaptation/assets/objects.py:303) 里注册。

本次 `chair` 的做法是：

1. 把 USD 放在：
   `active_adaptation/assets/objects/chair/object.usda`
2. 在 `objects.py` 里注册一个 `chair`
3. 在任务 yaml 里通过：
   `object_asset_name: "chair"`
   来引用它

注意：

- 单刚体对象推荐走 `RigidObjectCfg`
- 如果对象实际上是关节物体，才需要用 `ArticulationCfg`

---

## 5. 把自己的 motion 数据接进来

### 5.1 直接放入数据目录

假设你的数据已经整理成 HDMI 可读格式，把它放到：

```text
data/motion/g1/你的目录名/
  motion.npz
  meta.json
  annotation_summary.json
```

例如当前已有：

```text
data/motion/g1/chair/
  motion.npz
  meta.json
  annotation_summary.json
```

### 5.2 配置任务使用这份数据

可以在 yaml 里写死：

```yaml
command:
  data_path: data/motion/g1/chair
```

也可以运行时覆盖：

```bash
task.command.data_path=data/motion/g1/chair
```

---

## 6. 自定义对象任务的推荐配置方法

如果你的 motion 中对象 body 名和资产名不同，推荐单独建一个任务 yaml，像当前的：

- [custom_object_011_ref.yaml](C:/Users/17355/Desktop/VSCODE/hdmi1/cfg/task/G1/hdmi/custom_object_011_ref.yaml:1)

这个 yaml 适合“用 `chair` 资产去播放 motion 里名为 `custom_object_011` 的对象轨迹”。

推荐保留下面几项：

```yaml
command:
  root_body_name: "pelvis"
  object_asset_name: "chair"
  object_motion_name: "custom_object_011"
  object_body_name: "object"
```

如果你以后换了对象：

- `object_asset_name` 改成新资产名
- `object_motion_name` 改成新 `meta.json` 里的对象 body 名

---

## 7. 为什么要做 `chair1`

在这次数据里，原始 `chair/motion.npz` 的机器人和对象整体是歪的。原因不是仿真出错，而是：

- motion 里对象 `custom_object_011` 本身带有固定倾斜
- 代码会直接使用 motion 中的 world pose 回放

因此新增了一份“按椅子扶正”的数据目录：

- [data/motion/g1/chair1](C:/Users/17355/Desktop/VSCODE/hdmi1/data/motion/g1/chair1)

### 7.1 生成 `chair1` 的脚本

脚本：

- [scripts/upright_motion_by_object.py](C:/Users/17355/Desktop/VSCODE/hdmi1/scripts/upright_motion_by_object.py:12)

作用：

- 以对象第一帧姿态为基准
- 只消除对象的 `roll/pitch`
- 保留对象的 `yaw`
- 同时把整段机器人和对象的 world pose 一起旋转

### 7.2 生成命令

在仓库根目录运行：

```bash
python scripts/upright_motion_by_object.py \
  data/motion/g1/chair \
  data/motion/g1/chair1 \
  --object-body-name custom_object_011
```

生成后目录结构是：

```text
data/motion/g1/chair1/
  motion.npz
  meta.json
  annotation_summary.json
```

---

## 8. Play reference motion

`play.py` 用来播放 reference motion，不会保存视频。

### 8.1 Linux 服务器上播放 `chair1`

```bash
cd /home/boran/humanoid/newhdmi1
export PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES=0 python scripts/play.py \
  algo=ppo_roa_train \
  task=G1/hdmi/custom_object_011_ref \
  task.command.data_path=data/motion/g1/chair1 \
  +task.command.replay_motion=true \
  wandb.mode=disabled
```

### 8.2 如果想直接播放 `chair-sit`

```bash
cd /home/boran/humanoid/newhdmi1
export PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES=0 python scripts/play.py \
  algo=ppo_roa_train \
  task=G1/hdmi/chair-sit \
  task.command.data_path=data/motion/g1/chair1 \
  +task.command.replay_motion=true \
  wandb.mode=disabled
```

说明：

- `task.command.data_path=...`：切换到你的自定义数据目录
- `+task.command.replay_motion=true`：播放 reference motion
- `wandb.mode=disabled`：关闭 wandb

---

## 9. Render 视频

如果你要输出视频文件，用 `render.py`，不要用 `play.py`。

### 9.1 渲染 `chair1`

```bash
cd /home/boran/humanoid/newhdmi1
export PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES=0 python scripts/render.py \
  algo=ppo_roa_train \
  task=G1/hdmi/custom_object_011_ref \
  task.command.data_path=data/motion/g1/chair1 \
  +task.command.replay_motion=true \
  task.num_envs=1 \
  headless=true \
  eval_render=true \
  render_mode=rgb_array \
  wandb.mode=disabled
```

生成的视频通常会出现在：

```text
scripts/recording-*.mp4
```

---

## 10. 用自己的数据训练

### 10.1 直接用 `train.py`

```bash
cd /home/boran/humanoid/newhdmi1
export PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  algo=ppo_roa_train \
  task=G1/hdmi/custom_object_011_ref \
  task.command.data_path=data/motion/g1/chair1 \
  wandb.mode=online \
  wandb.project=hdmi \
  exp_name=chair1_custom
```

### 10.2 用 `mytrain.py`

`mytrain.py` 支持 `--data-dir`，会自动改写 `task.command.data_path`。

```bash
cd /home/boran/humanoid/newhdmi1
export PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES=0 python scripts/mytrain.py \
  --data-dir data/motion/g1/chair1 \
  algo=ppo_roa_train \
  task=G1/hdmi/custom_object_011_ref \
  wandb.mode=online \
  wandb.project=hdmi \
  exp_name=chair1_custom
```

---

## 11. 常见问题

### 11.1 `ModuleNotFoundError: No module named 'scripts'`

先设置：

```bash
export PYTHONPATH=$PWD
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

### 11.2 `wandb.mode=disable` 不生效

正确写法是：

```bash
wandb.mode=disabled
```

### 11.3 `play.py` 为什么不出视频

因为：

- `play.py` 只负责播放
- `render.py` 才负责导出视频

### 11.4 `headless=false` 时报显示或 GPU 错误

如果服务器没有桌面环境、没有可用显示或 Vulkan 渲染环境，会出现：

- `Failed to open display`
- `No device could be created`
- `GLFW initialization failed`

这时：

- 看 reference motion 可以用 `headless=true`
- 真正开窗口需要可用显示和 GPU 渲染环境

### 11.5 物体路径或 contact filter 报错

如果你在用旧代码，可能会碰到：

- 找不到 `chair.usd`
- 找不到 `/World/envs/env_*/chair/object`

当前仓库已修正为：

- `chair` 资产入口使用 `object.usda`
- 单刚体对象兼容直接挂在 `/chair` 根路径

如果服务器上还报这些问题，先同步最新代码：

```bash
git pull origin main
```

---

## 12. 推荐的完整顺序

建议按下面顺序操作：

1. 准备自己的 `motion.npz + meta.json`
2. 把对象 USD 放到 `active_adaptation/assets/objects/<your_object>/`
3. 在 `active_adaptation/assets/objects.py` 注册对象
4. 新建一个任务 yaml，写好：
   - `data_path`
   - `object_asset_name`
   - `object_motion_name`
   - `object_body_name`
5. 先用 `play.py + replay_motion=true` 检查 reference motion
6. 如果整体姿态歪，先用 `upright_motion_by_object.py` 生成一份扶正后的新目录
7. 再用新目录做 `play`
8. 用 `render.py` 导出视频
9. 最后再启动训练

---

## 13. 本次示例最常用的命令

### 13.1 生成扶正后的 `chair1`

```bash
python scripts/upright_motion_by_object.py \
  data/motion/g1/chair \
  data/motion/g1/chair1 \
  --object-body-name custom_object_011
```

### 13.2 播放 `chair1`

```bash
cd /home/boran/humanoid/newhdmi
export PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES=0 python scripts/play.py \
  algo=ppo_roa_train \
  task=G1/hdmi/custom_object_011_ref \
  task.command.data_path=data/motion/g1/chair1 \
  +task.command.replay_motion=true \
  wandb.mode=disabled
```

### 13.3 渲染 `chair1` 视频

```bash
cd /home/boran/humanoid/newhdmi1
export PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES=0 python scripts/render.py \
  algo=ppo_roa_train \
  task=G1/hdmi/custom_object_011_ref \
  task.command.data_path=data/motion/g1/chair1 \
  +task.command.replay_motion=true \
  task.num_envs=1 \
  headless=true \
  eval_render=true \
  render_mode=rgb_array \
  wandb.mode=disabled
```

### 13.4 用 `chair1` 训练

```bash
cd /home/boran/humanoid/newhdmi1
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=0

python scripts/mytrain.py \
  --data-dir data/motion/g1/chair1 \
  algo=ppo_roa_train \
  task=G1/hdmi/custom_object_011_ref \
  task.num_envs=64 \
  total_frames=20000000 \
  wandb.mode=online \
  wandb.project=hdmi \
  exp_name=chair1_custom_10k
```

---

## 14. 当前这套示例最终对应关系

- 对象资产名：`chair`
- 对象 USD：`active_adaptation/assets/objects/chair/object.usda`
- 原始数据目录：`data/motion/g1/chair`
- 扶正后数据目录：`data/motion/g1/chair1`
- motion 中对象 body 名：`custom_object_011`
- 自定义任务 yaml：`cfg/task/G1/hdmi/custom_object_011_ref.yaml`

如果以后你换成别的对象，通常只要同步改这四类内容：

- 对象 USD 路径
- `object_asset_name`
- `object_motion_name`
- `task.command.data_path`
