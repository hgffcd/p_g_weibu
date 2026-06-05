# Implementation Plan

## 1. 文件结构

```text
.
├── README.md
├── requirements.txt
├── config.yaml
├── assumptions.md
├── main.py
├── train.py
├── evaluate.py
├── envs/
│   ├── __init__.py
│   └── encirclement_env.py
├── agents/
│   ├── __init__.py
│   ├── apf_guide.py
│   └── simple_actor.py
├── algorithms/
│   ├── __init__.py
│   ├── regulators.py
│   └── mra_rlec.py
├── utils/
│   ├── __init__.py
│   ├── config.py
│   └── geometry.py
└── tests/
    ├── test_env.py
    ├── test_reward.py
    ├── test_model.py
    └── test_training_smoke.py
```

## 2. 每个文件的作用

- `README.md`：说明论文复现范围、安装、运行训练和评估命令。
- `requirements.txt`：列出最小依赖。当前最小运行只需要 `numpy` 和 `PyYAML`；完整 neural PPO 需要补充 `torch`。
- `config.yaml`：集中配置环境、奖励、guide policy、regulators、训练参数。
- `assumptions.md`：记录论文缺失信息和工程假设。
- `main.py`：命令入口，分发 `train` 或 `evaluate`。
- `train.py`：最小训练或 rollout smoke loop。
- `evaluate.py`：运行若干 episode 并输出成功、碰撞、平均奖励等指标。
- `envs/encirclement_env.py`：二维 EMOCA 环境。
- `agents/apf_guide.py`：公式 (7)-(10) 的 APF guide policy。
- `agents/torch_networks.py`：表 II 对应的 PyTorch recurrent actor/centralized critic。
- `agents/simple_actor.py`：NumPy fallback actor/critic forward，用于 smoke test。
- `algorithms/regulators.py`：公式 (11)-(14) 的 curriculum regulators。
- `algorithms/rollout_buffer.py`：Algorithm 1 第 7-9 行的轨迹存储与 GAE。
- `algorithms/ppo.py`：公式 (6) PPO clipped objective 和 critic MSE loss。
- `algorithms/mra_rlec.py`：Algorithm 1 的可运行训练流程。
- `utils/config.py`：YAML 配置读取。
- `utils/geometry.py`：距离、角度、裁剪、邻居排序等几何函数。
- `tests/`：标准库 `unittest` 测试，不依赖 pytest。

## 3. 每个核心类的职责

- `EncirclementEnv`
  - 管理 pursuers、target、obstacles 的状态。
  - 实现公式 (1) 的动力学积分。
  - 实现公式 (2) 围捕判定。
  - 实现公式 (15)-(18) reward。
  - 实现公式 (19) local observation。

- `APFGuidePolicy`
  - 根据公式 (7)-(10) 计算每个 pursuer 的 guide acceleration。
  - 负责动作裁剪。

- `RecurrentActor`
  - 接收 `[N, obs_dim]` 观测和 GRU hidden state。
  - 输出 Gaussian action distribution、动作和 log probability。
  - 对应第 IV-C.3 节和表 II。

- `RecurrentCritic`
  - 接收扁平化全局状态和 GRU hidden state。
  - 输出 centralized scalar value。
  - 对应第 IV-C.3 节和表 II。

- `TorchMRARLECTrainer`
  - 组织 Algorithm 1 的 episode loop。
  - 调用 regulators 决定 guide policy 步数、围捕偏差和障碍物半径。
  - 收集轨迹、计算 GAE、执行 PPO update。

## 4. 每个核心函数的输入输出

- `EncirclementEnv.reset(seed=None) -> (observations, state)`
  - 输入：随机种子。
  - 输出：局部观测矩阵 `[N, 24]` 和全局状态字典。

- `EncirclementEnv.step(actions) -> (observations, rewards, done, info)`
  - 输入：动作矩阵 `[N, 2]`。
  - 输出：下一观测 `[N, 24]`，奖励 `[N]`，终止布尔值，诊断信息。

- `EncirclementEnv.compute_rewards() -> rewards`
  - 输入：内部状态。
  - 输出：GERD 奖励 `[N]`。

- `APFGuidePolicy.act(env) -> actions`
  - 输入：环境实例或环境状态。
  - 输出：guide action `[N, 2]`。

- `jsrl_regulator(x, k_s) -> float`
  - 输入：JSRL 进度 `x in [0,1]`。
  - 输出：guide step 比例。

- `MRARLECTrainer.train() -> history`
  - 输入：配置。
  - 输出：每 episode 总奖励、终止原因、guide steps。

## 5. 数据流

```text
config.yaml
  -> load_config()
  -> EncirclementEnv / APFGuidePolicy / Actor / Trainer
  -> trainer.train()
  -> env.reset()
  -> observations
  -> guide policy or actor outputs actions
  -> env.step(actions)
  -> rewards, done, info
  -> history
```

## 6. 训练流程或算法运行流程

最小训练：

1. 读取配置。
2. 初始化环境、APF guide policy、简单 actor。
3. 每个 episode 根据公式 (11)-(14) 更新 regulators。
4. 前 `M^g` 步使用 guide policy，后续使用 actor。
5. 执行环境 step，累计 reward。
6. 使用 GAE 计算 advantage。
7. 使用公式 (6) 的 PPO clipped loss 更新 actor，使用 MSE value loss 更新 critic。
8. 打印每个 episode 的 reward、success、collision、steps、actor loss、critic loss。

完整论文训练仍需补充多线程采样、原始随机场景分布和 ROS2/Gazebo 接口。

## 7. 测试计划

- reset 测试：验证观测形状、状态字典字段和初始速度。
- step 测试：验证动作输入后位置和速度更新，输出格式正确。
- reward 测试：人工设置近似围捕状态，验证 global/local/step reward 分支。
- model forward 测试：验证 actor 输出 `[N, 2]`，critic 输出标量。
- training smoke test：运行 2 个 episode、每个 episode 少量 step，验证无异常、history 字段完整。

测试使用标准库 `unittest`，命令：

```bash
python -m unittest discover -s tests
```

## 8. 可能无法完全复现的部分

- 原论文完整 rMAPPO 训练曲线、表 IV/V 数值结果。
- Gazebo + ROS2 全向车仿真。
- PFP、AP、MCP 目标逃逸策略的完整训练组合。
- 原始障碍物生成分布、训练线程数、随机种子、归一化和 optimizer 细节。
- critic 全局输入和 RNN hidden state 的原始实现细节。

## 9. 工程假设

- 默认 `N = 5`，`M = 6`，障碍物坐标由配置给出。
- 初始 pursuers 沿 x 轴直线排列，target 位于前方。
- 无碰撞时 collision penalty 为 0，碰撞时为 `R_l = -3`。
- 最近障碍物按 pursuer 到障碍物中心距离减半径选择。
- 观测中的若干角速度导数项在最小实现中置零。
- 本机默认 Python 缺少 PyTorch，因此最小可运行版本用 NumPy 实现 actor/critic forward；`requirements.txt` 仍说明完整版本应安装 PyTorch。
