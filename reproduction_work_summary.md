# 论文代码复现阶段总结与后续提示词

本文档总结《A Policy-Guided Reinforcement Learning Method for Encirclement Control in Multiobstacle Environment》代码复现过程中已经完成的工作、遇到的问题、解决方案、强化学习参数和奖励调整，以及后续继续工作的提示词。

## 1. 当前复现定位

当前工程不是原论文完整仿真平台复刻，而是论文方法在二维平面圆形智能体环境中的最小工程复现。

已经实现的论文方法要素：

- 二维 EMOCA 环境：多追捕者、多静态圆形障碍物、目标实体。
- 追捕者二阶积分器动力学：位置、速度、加速度裁剪。
- 围捕成功条件：距离误差和相邻夹角误差满足论文 Eq. (2)。
- APF guide policy：追捕者间斥力、障碍物斥力、目标吸引力、阻尼项。
- JSRL guide-step curriculum：训练前期更多使用 guide policy，后期逐渐减少。
- angle / distance / obstacle curriculum regulators。
- GERD 奖励结构：全局成功奖励、局部完成奖励、step reward、碰撞惩罚。
- 局部观测：左邻居、右邻居、目标、最近障碍物。
- rMAPPO/PPO 风格训练：共享 actor、集中 critic、GAE、PPO clipped objective。
- 服务器向量化训练：4 张 GPU 各自启动独立 seed 训练。
- 可视化：`visualize.py` 支持 `.html`、`.gif`、`.mp4`。

当前简化点：

- 不实现 ROS2/Gazebo、模型车、轮速转换和车辆动力学。
- 当前主配置目标策略是 `static`，即目标不移动。
- 当前训练结果主要验证固定场景，不是随机化多场景泛化。
- 当前 checkpoint 策略是 `guide + residual actor`，不是纯 actor-only。
- AP/MCP 目标逃逸策略尚未完整实现。

## 2. 当前关键配置

主要配置文件：

```text
config_vector_server.yaml
```

当前成功配置核心参数：

```yaml
environment:
  num_pursuers: 5
  max_steps: 200
  dt: 0.1
  capture_distance: 1.0
  delta_alpha: 0.3
  delta_distance: 0.1
  pursuer_radius: 0.15
  target_radius: 0.15
  safety_buffer: 0.05
  max_velocity: 0.5
  target_max_velocity: 0.15
  max_acceleration: 0.5
  target_policy: static
  target_initial_position: [0.0, 3.5]
```

当前 guide policy 关键参数：

```yaml
guide_policy:
  mode: environment_py
  kr: 4.5
  ko: 4.0
  ka: 2.0
  kb: 1.5
  target_gain: 2.0
  target_velocity_gain: 1.5
  inner_target_gain_scale: 20.0
  obstacle_extra_margin: 0.3
  safety_filter: true
  safety_gain: 6.0
  safety_margin: 0.3
  safety_damping: 1.0
  formation_correction: false
```

当前训练关键参数：

```yaml
training:
  policy_mode: residual
  residual_scale: 0.1
  gamma: 0.985
  gae_lambda: 0.95
  ppo_clip: 0.15
  ppo_epochs: 4
  batch_size: 2048
  learning_rate_base: 0.0001
  learning_rate_fine_tune: 0.00015
  entropy_coef: 0.0002
  value_coef: 0.5
  bc_coef: 2.0
  max_grad_norm: 0.25
```

当前向量化训练参数：

```yaml
vector_training:
  num_envs: 16
  rollout_length: 64
  updates: 2000
  curriculum_progress: updates
  eval_interval: 50
  eval_episodes: 20
```

## 3. 复现过程中遇到的问题与解决

### 3.1 PDF 信息不完整，无法直接照抄全部实验

问题：

- 论文没有给出所有环境初始坐标、障碍物完整采样规则、critic 输入拼接细节、GAE 参数、batch 组织方式等。
- 表 IV/V 的完整数值无法从工程层面直接复现。

处理：

- 先输出 `paper_analysis.md`、`algorithm_formulation.md`、`algorithm_pseudocode.md`、`implementation_plan.md`。
- 所有缺失细节写入 `assumptions.md`。
- 采用最小工程假设实现可运行平面环境。

### 3.2 `environment.py` 不是完整可运行论文代码

问题：

- 用户新增 `environment.py`。
- 分析后发现它是 MPE/onpolicy 风格环境封装，依赖 `gym`、`multi_discrete`、`onpolicy.global_var`、callbacks、world/policy_agents。
- 不能作为当前工程的独立主环境直接运行。

处理：

- 不直接依赖 `environment.py`。
- 提取其中有价值的实现思想：
  - `policy_u` 风格 guide policy；
  - `limit_action_inf_norm`；
  - 类似 `set_JS_curriculum` 的课程变化。
- 将这些迁移到：
  - `agents/apf_guide.py`
  - `algorithms/regulators.py`
  - `envs/encirclement_env.py`

### 3.3 PyTorch 分布 `scale` 出现 NaN

问题：

服务器训练时报错：

```text
ValueError: Expected parameter scale ... Normal(... scale=...) ... found invalid values: nan
```

原因：

- actor 的 `log_std` 或隐藏状态数值不稳定，导致 `std = exp(log_std)` 出现 NaN。

处理：

- 在 `agents/torch_networks.py` 中增加数值保护：
  - `torch.nan_to_num` 处理 observation、hidden、mean、log_std；
  - `log_std` clamp 到 `[LOG_STD_MIN, LOG_STD_MAX]`；
  - `std.clamp_min(1e-6)`；
  - 新增 `clamp_distribution_parameters()`。

### 3.4 原始训练切换到 actor 后成功率很低

问题：

- 早期实现中，JSRL 后期 guide 消失后，actor 独立控制。
- 训练到较多 episode 后成功率仍低，甚至碰撞或超时。

原因：

- 当前环境奖励稀疏，APF guide 可以完成部分接近，但 actor-only 很难稳定学到完整围捕几何。
- 围捕成功条件同时要求距离和角度，单纯 GERD 奖励对角度分布信号不足。

处理：

- 将训练模式改为 residual：

```text
action = guide_action + residual_scale * actor_action
```

- 当前 `residual_scale = 0.1`。
- pretrain 阶段让 actor 学习零残差，避免破坏 guide。
- PPO 训练时 guide step 中 actor objective 不强行学习 guide 大动作，而是在 residual 模式下学习修正量。

### 3.5 GPU 利用率低，单卡训练慢

问题：

- 单卡训练 GPU 占用低，CPU rollout 成为瓶颈。

处理：

- 实现 `VectorizedEncirclementEnv` 和 `VectorMAPPOTrainer`。
- 每张 GPU 启动一个独立进程，各自使用不同 seed 和 checkpoint/log 目录。
- 脚本：

```text
scripts/launch_4gpu_vector.sh
```

- 每张卡独立运行：

```text
config_vector_gpu0.yaml
config_vector_gpu1.yaml
config_vector_gpu2.yaml
config_vector_gpu3.yaml
```

这些 GPU 配置由脚本自动从 `config_vector_server.yaml` 生成，不需要长期保留。

### 3.6 4 GPU 启动后看不到训练进度

问题：

- 用户希望终端直接看到训练是否进行、是否结束。
- 早期脚本只后台启动，不方便监控。

处理：

- 更新 `scripts/launch_4gpu_vector.sh`：
  - 前台循环监控；
  - 每隔 `MONITOR_INTERVAL` 秒打印每张 GPU 最新 `update=` 行；
  - 支持 `STARTUP_TIMEOUT`，如果长时间没有 `update=1` 自动停止，避免服务器卡住。

### 3.7 训练过程遇到 `pretrain_actor.pt` 找不到

问题：

```text
FileNotFoundError: checkpoints/pretrain_actor.pt
```

原因：

- 启动训练前没有先运行 pretrain，或路径不一致。

处理：

- 训练流程改为先执行：

```text
python main.py pretrain --config config_vector_server.yaml
```

- 再启动 vector train。

### 3.8 成功率为 0，全部碰撞

问题：

- 早期服务器结果表现为：

```text
success_rate = 0
collision_rate = 1
obstacle_collision_rate = 1
```

原因：

- guide 策略接近目标时容易穿过障碍安全区。
- 环境目标位置和障碍物布局过于接近，围捕圆与障碍安全区冲突或近似冲突。
- guide 的障碍斥力和安全过滤不够稳定。

处理：

- 增加安全过滤：

```yaml
safety_filter: true
safety_gain: 6.0
safety_margin: 0.3
safety_damping: 1.0
```

- 增加 obstacle extra margin：

```yaml
obstacle_extra_margin: 0.3
```

- 在 `collision_detail()` 中区分：
  - pursuer collision；
  - obstacle collision；
  - collision distance；
  - collision threshold。

### 3.9 成功率为 0，全部超时

问题：

- 改完碰撞后，变成不碰撞但全部超时。

诊断：

- 新增 `experiments/diagnose_guide.py`，直接测试 guide policy。
- 结果表明 guide 本身无法在原目标位置完成严格 Eq. (2) 围捕。

核心发现：

- 原先目标位置 `[0.0, 2.2]` 离障碍过近。
- 目标 capture ring 几乎贴近障碍安全区。
- 这是几何可行性问题，不是单纯 PPO 参数问题。

处理：

- 将目标初始位置改为：

```yaml
target_initial_position: [0.0, 3.5]
```

- 写入 `assumptions.md`，说明论文没有给出精确场景坐标，因此当前采用可行化工程假设。

结果：

- guide 诊断成功率恢复到 1.0。
- 后续训练也能稳定成功。

### 3.10 尝试 formation correction 后效果变差

问题：

- 为解决角度误差，曾加入 ring-slot formation correction。
- 直接启用后容易把智能体推向障碍或引入额外震荡。

处理：

- 增加 delayed/ramped/gated formation logic：
  - `formation_start_step`
  - `formation_ramp_steps`
  - `formation_obstacle_gate_margin`
  - `formation_slot_clearance_margin`

最终决策：

- 在当前可行几何配置下关闭：

```yaml
formation_correction: false
```

原因：

- guide + residual 已能满足成功条件；
- 额外 formation term 增加风险，收益不明显。

### 3.11 奖励信号不足，加入最小 shaping

问题：

- 原论文 GERD 奖励主要包含 formation aggregate 和 distance reward，但在当前简化环境里对严格角度围捕信号不足。

处理：

- 在 `EncirclementEnv.compute_rewards()` 中加入工程 shaping：

```yaml
angle_shaping_weight: 0.8
distance_shaping_weight: 0.6
timeout_penalty: -2.0
```

- shaping 使用：
  - `encirclement_errors()` 中的平均距离误差；
  - 平均角度误差。

注意：

- 这是工程补充，不是论文原始明确公式。
- 已写入 `assumptions.md`。

### 3.12 训练结果 500 回合完全相同

现象：

- `vector_eval_500` 中每张 GPU 的 500 个 episode 结果完全一致。
- 例如同一 GPU 的 steps、final_distance_error、final_angle_error 每回合相同。

原因：

- 当前环境 `reset(seed=episode)` 没有真正随机化目标、障碍物或追捕者初始位置。
- 因此 500 回合是同一固定场景重复评估。

结论：

- 当前结果证明固定场景稳定成功。
- 不能证明多场景泛化。

### 3.13 可视化文件混淆 guide 和 checkpoint

问题：

- 生成过 `guide_episode0.gif`，它展示的是 APF guide，不是训练后 checkpoint。

处理：

- 新增 `visualize.py`，支持：
  - `--policy guide`
  - `--policy controller`
  - `--policy checkpoint`

训练后模型可视化应使用：

```text
--policy checkpoint
--checkpoint checkpoints_vector/gpu1/mra_rlec_best.pt
--output results/visualization/gpu1_episode0.gif
```

### 3.14 可视化 status 文本有误导

问题：

- GIF 第 0 帧显示 `success=True`，这是终止结果被显示到所有帧，而不是第 0 步已经成功。

后续应修复：

- rollout 时保存每一帧的 info 或 step status。
- animation update 中按 frame 显示当时状态，而不是最终 info。

### 3.15 MP4 生成依赖问题

问题：

服务器执行：

```text
ffmpeg -version
```

出现：

```text
ffmpeg: error while loading shared libraries: libiconv.so.2
```

原因：

- `ffmpeg` 能找到，但动态库 `libiconv.so.2` 缺失。

处理建议：

- 在 `phf_env` 中安装或重装：

```text
conda install -c conda-forge libiconv ffmpeg
```

- 若仍失败，使用 GIF，不依赖 ffmpeg。

## 4. 当前结果

已解压的 `mra_rlec_server_artifacts_500` 中，四张 GPU 的 best checkpoint 在固定场景 500 回合评估结果：

```text
success_rate = 1.0
collision_rate = 0.0
timeout_rate = 0.0
mean_encirclement_time = 16.225 s
mean_final_distance_error = 0.0127
mean_final_angle_error = 0.1164
mean_min_obstacle_clearance = 0.2650
mean_min_pursuer_clearance = 0.3149
```

每张 GPU：

```text
gpu0: success=1.0, collision=0.0, timeout=0.0, encirclement_time=15.1s
gpu1: success=1.0, collision=0.0, timeout=0.0, encirclement_time=17.4s
gpu2: success=1.0, collision=0.0, timeout=0.0, encirclement_time=15.8s
gpu3: success=1.0, collision=0.0, timeout=0.0, encirclement_time=16.6s
```

解释：

- 当前固定场景围捕成功。
- 不是任意目标点、任意障碍物、任意追捕者数量的泛化成功。
- 当前模型执行方式是 residual：

```text
action = guide_action + 0.1 * actor_action
```

## 5. 与论文结果的差距

论文不是只做静止目标或单一场景。论文实验包含：

- 移动 target；
- 多种 target escape policies：PFP、GP、AP、MCP；
- 不同追捕者数量：N=3、4、5、6；
- 与多种 baseline 对比；
- 数值仿真和 Gazebo/ROS2 半物理仿真；
- 指标包括 success rate、danger rate、encirclement time、average step reward、distance/angle error、collision distance 等。

当前工程还缺：

- 移动目标正式训练；
- PFP/Greedy 多场景评估；
- AP/MCP 完整实现；
- 随机化 reset；
- 固定场景与随机场景分离评估；
- 不同 N 的模型或可变智能体网络；
- 与 baseline 的统一对比实验；
- checkpoint GIF 结果确认；
- 可视化逐帧 status 修复。

## 6. 下一阶段建议路线

### 阶段 1：补齐 checkpoint 可视化

目标：

- 生成 `gpu1_episode0.gif`，确认训练后策略视觉效果。
- 修复可视化状态文字逐帧显示。

### 阶段 2：固定场景 + 移动目标

目标：

- 将 `target_policy` 从 `static` 改为 `pfp` 或 `greedy`。
- 重新训练。
- 比较 static / pfp / greedy 下：
  - success rate；
  - collision rate；
  - timeout rate；
  - encirclement time；
  - final distance error；
  - final angle error。

### 阶段 3：随机目标位置

目标：

- 修改 `EncirclementEnv.reset()` 支持：

```yaml
target_randomization:
  enabled: true
  x_range: [-1.5, 1.5]
  y_range: [2.5, 4.5]
```

- 增加可行性检查：
  - 目标 capture ring 不与障碍安全区重叠；
  - 初始 pursuer 不碰撞；
  - target 不在障碍内。

### 阶段 4：随机追捕者初始队形

目标：

- 不再固定横排。
- 支持从区域或若干模板队形采样初始位置。

### 阶段 5：随机障碍扰动

目标：

- 在固定障碍基础上增加轻微位置扰动和半径扰动。
- 加可行性重采样。

### 阶段 6：可变追捕者数量

短期方案：

- N=3、4、5、6 分别训练不同模型。

长期方案：

- 改成 GNN 或 attention policy，支持可变数量智能体。

## 7. 后续继续工作的提示词

下面提示词可直接给后续代码助手使用。

```text
你现在继续接手一个论文复现工程，论文是《A Policy-Guided Reinforcement Learning Method for Encirclement Control in Multiobstacle Environment》。

当前工程目录是 p_g_weibu，已经实现了二维平面圆形智能体版本的 MRA-RLEC 最小复现。不要重新从零开始，先阅读以下文件：

1. paper_analysis.md
2. algorithm_formulation.md
3. algorithm_pseudocode.md
4. implementation_plan.md
5. assumptions.md
6. implementation_notes.md
7. reproduction_work_summary.md
8. config_vector_server.yaml
9. envs/encirclement_env.py
10. agents/apf_guide.py
11. algorithms/vector_mappo.py
12. visualize.py

当前已完成：

- 二维平面 EMOCA 环境；
- APF guide policy；
- JSRL curriculum；
- GERD reward + 工程 shaping；
- rMAPPO/PPO 风格训练；
- residual policy mode；
- 4 GPU 独立 seed 向量化训练；
- fixed scene 下 500 回合评估 100% 成功；
- GIF/HTML/MP4 可视化入口。

当前关键配置：

- num_pursuers = 5
- target_policy = static
- target_initial_position = [0.0, 3.5]
- policy_mode = residual
- residual_scale = 0.1
- angle_shaping_weight = 0.8
- distance_shaping_weight = 0.6
- timeout_penalty = -2.0
- guide_policy.mode = environment_py
- formation_correction = false

当前结果边界：

- 成功结果只证明固定目标点、固定障碍物、5 个追捕者、静止目标下稳定围捕；
- 500 episode 评估是确定性重复，不是随机场景泛化；
- 当前 checkpoint 推理是 guide + residual actor，不是纯 actor-only；
- 当前 GIF 可能只有 guide_episode0.gif，需要生成 checkpoint 的 gpu1_episode0.gif；
- 可视化状态文字目前显示的是最终 info，可能误导，需要改成逐帧状态。

请继续完成下一阶段工作，优先级如下：

第一步：修复 visualize.py，使动画每一帧显示当前 step 的状态，而不是所有帧都显示最终 success/collision/timeout。

第二步：生成并分析 checkpoint 可视化，使用：
python visualize.py --config config_vector_server.yaml --policy checkpoint --checkpoint checkpoints_vector/gpu1/mra_rlec_best.pt --episode 0 --output results/visualization/gpu1_episode0.gif

第三步：加入移动目标复现实验。先使用已有 target_policy = pfp 或 greedy，在固定障碍场景下重新训练和评估，不要同时引入随机化。

第四步：新增 fixed_eval 和 randomized_eval 的区别。fixed_eval 保持当前固定场景；randomized_eval 后续用于随机目标点、随机初始队形和随机障碍扰动。

第五步：实现 target 位置随机化，但必须加入可行性检查，避免 capture ring 与障碍安全区冲突。不要再使用不可行目标点 [0.0, 2.2]。

第六步：保留所有论文未明确说明的工程假设，并写入 assumptions.md。不要声称已经完整复现原论文全部结果。

开发原则：

- 不要删掉核心源码；
- 不要把训练产物当作源码提交；
- 所有关键改动要能通过 python -m unittest discover -s tests；
- 服务器环境是 phf_env，Python 3.10，torch 2.6.0，CUDA 12.4，numpy 1.26.4，PyYAML 6.0.3，matplotlib 3.10.0，pillow 11.3.0；
- MP4 需要 ffmpeg，如果 ffmpeg 依赖错误则使用 GIF；
- 优先保证固定场景不回退，再逐步增加移动目标和随机场景。
```

