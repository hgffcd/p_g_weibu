# Implementation Notes

## 1. 已实现的论文内容

- 二维 EMOCA 环境：pursuer、target、static circular obstacles。来源：第 II-A 节。
- 二阶积分器动力学和速度/加速度裁剪。来源：第 II-A 节、公式 (1)。
- 围捕条件判断。来源：第 II-A 节、定义 2、公式 (2)。
- APF guide policy。来源：第 IV-A 节、公式 (7)-(10)。
- JSRL、角度、距离和障碍物半径 curriculum regulators。来源：第 IV-A/B 节、公式 (11)-(14)。
- GERD reward。来源：第 IV-C.1 节、公式 (15)-(18)。
- 双向邻居观测和 24 维局部观测。来源：第 IV-C.2 节、公式 (19)。
- Algorithm 1 的 PyTorch 训练骨架。
- 表 II 对应的 recurrent actor 和 centralized recurrent critic。
- GAE rollout buffer 和公式 (6) PPO clipped objective。
- PFP 和 GP 目标逃逸策略。来源：第 V-D 节、公式 (20)-(21)。
- reset、step、reward、model forward、training loop smoke tests。

## 2. 未实现的论文内容

- 多线程采样和与论文完全一致的 batch 组织。
- ROS2/Gazebo 全向车仿真。
- AP、MCP target escape policies 的完整实现和 Monte Carlo 实验。
- 表 IV、表 V 的完整实验复现。
- 原论文训练规模：Base `7.68e6` steps、Fine Tune `5.12e6` steps。

## 3. 与原论文不同的地方

- PPO 使用平均 team reward 和 centralized scalar value；论文未给出所有 advantage/value target 工程细节。
- guide policy 动作的 PPO old log probability 由当前 actor 对该动作评估得到；论文未说明 guide-generated action 的 off-policy 修正方式。
- `config.yaml` 使用 JSON-compatible YAML，以便在没有 PyYAML 的环境中运行。
- 默认 target policy 为 static；已实现 PFP/GP，AP/MCP 未实现。

## 4. 工程假设

详见 `assumptions.md`。关键假设包括：初始位置、障碍物配置、无碰撞时 `R_l = 0`、观测导数项置零、critic 全局状态简化。

## 5. 如何运行代码

使用 `rl_env`：

```bash
conda activate rl_env
python main.py train --config config.yaml
python main.py evaluate --config config.yaml --episodes 2
```

## 6. 如何测试代码

```bash
python -m unittest discover -s tests
```

上一轮最小版本测试结果：

```text
Ran 6 tests in 0.070s
OK
```

上一轮最小版本训练入口 smoke 输出：

```text
episode=1 reward=-14.970 steps=24 guide_steps=12 success=False collision=True timeout=False
episode=2 reward=-11.736 steps=18 guide_steps=0 success=False collision=True timeout=False
episode=3 reward=-11.171 steps=17 guide_steps=0 success=False collision=True timeout=False
```

评估入口 smoke 输出：

```text
episodes=2 mean_reward=-11.171 success_rate=0.000 collision_rate=1.000
```

## 7. 后续需要补充的数据、参数或实验

- 原论文完整障碍物布局、初始化随机分布、训练 seed 和归一化细节。
- PFP、GP、AP、MCP 逃逸策略完整实现。
- 与 ECBVC、PFEC、FBEC、MADDPG、MATD3 等方法的统一实验脚本。
- Gazebo/ROS2 模型、消息接口和全向车逆运动学控制层。
- 长时间训练以接近论文 Base `7.68e6` steps 和 Fine Tune `5.12e6` steps 的规模。
