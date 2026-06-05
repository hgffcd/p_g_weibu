# Policy-Guided Multi-Agent Encirclement Control

这是论文 **A Policy-Guided Reinforcement Learning Method for Encirclement Control in Multiobstacle Environment** 的代码复现工程。

本仓库不是原论文的模型车/ROS/Gazebo 完整仿真复刻，而是按照论文核心思想实现的二维平面圆形环境版本：追捕者、目标和障碍物均建模为圆，重点复现“策略引导 + 强化学习围捕”的算法流程。

## 当前实现效果

已实现并可运行的内容：

- 二维多追捕者围捕环境，包含追捕者、目标、障碍物、碰撞检测、围捕判定。
- 策略引导机制：APF/environment.py 风格引导策略与 Stage-2 channel guide。
- residual MAPPO 训练：以引导策略为 nominal action，神经网络学习残差修正。
- 引导比例衰减训练流程，近似论文中的 policy-guided reinforcement learning。
- 集中式 critic、共享 actor、多智能体 rollout、GAE/PPO 更新。
- 多 GPU 服务器启动脚本，每张 GPU 独立跑一组随机种子。
- 评估指标：成功率、碰撞率、超时率、围捕时间、最终距离误差、角度误差、障碍物安全距离、追捕者间安全距离、通信消息量估计。
- GIF/HTML/MP4 可视化，其中 GIF 不需要系统 ffmpeg。

在当前 `v3 channel` 场景中，5 个追捕者可在多障碍二维平面中完成围捕。4 个服务器 checkpoint 的 300 episode 评估结果均为：

| checkpoint | success_rate | collision_rate | timeout_rate | encirclement_time |
|---|---:|---:|---:|---:|
| gpu0 best | 1.000 | 0.000 | 0.000 | 16.718 |
| gpu1 best | 1.000 | 0.000 | 0.000 | 16.726 |
| gpu2 best | 1.000 | 0.000 | 0.000 | 16.583 |
| gpu3 best | 1.000 | 0.000 | 0.000 | 18.224 |

速度压力测试中，channel guide 在 `target_speed = 0.00, 0.05, 0.10, 0.15, 0.20, 0.25` 下均达到 `success_rate = 1.0`，对应摘要见 [results/reference_metrics](results/reference_metrics)。

代表性围捕效果如下：

![stage2 encirclement result](assets/stage2_gpu1_episode0.gif)

更多 GIF 位于 [assets](assets)。

## 不能实现或不能保证的效果

当前版本不能保证：

- 复现原论文的真实模型车、底盘动力学、ROS/Gazebo 或物理平台实验。
- 对任意目标点、任意障碍物布局、任意目标速度都稳定成功。
- 对任意数量追捕者自动泛化。
- 严格达到原论文所有实验表格数值，因为论文部分工程参数、真实仿真细节和平台细节缺失。

额外构造的 `hard_v1` 压力测试没有成功解决。4 张卡 500 episode 评估平均约：

| metric | value |
|---|---:|
| success_rate | 0.3525 |
| collision_rate | 0.6475 |
| timeout_rate | 0.0000 |

该结果说明困难随机场景的主要失败原因是碰撞，不是训练没有跑完。继续提高泛化能力需要改进安全引导、碰撞屏障或分阶段随机化课程，而不是单纯增加训练轮数。

## 文件结构

```text
agents/        引导策略、启发式控制器、actor/critic 网络
algorithms/    PPO/MAPPO、rollout buffer、课程调节器
envs/          基础二维围捕环境
stage2/        当前主线：随机目标、channel guide、训练/评估/可视化脚本
utils/         配置、几何、指标工具
tests/         基础单元测试和 smoke test
tests_stage2/  Stage-2 环境测试
assets/        代表性 GIF 结果图
checkpoints/   代表性已训练 checkpoint
results/       精简后的参考评估指标
```

论文解析和工程设计文档：

- [paper_analysis.md](paper_analysis.md)
- [algorithm_formulation.md](algorithm_formulation.md)
- [algorithm_pseudocode.md](algorithm_pseudocode.md)
- [implementation_plan.md](implementation_plan.md)
- [assumptions.md](assumptions.md)
- [reproduction_work_summary.md](reproduction_work_summary.md)
- [STAGE2_WORKFLOW.md](STAGE2_WORKFLOW.md)

## 环境依赖

服务器环境已按 `phf_env` 适配。已知可用版本包括：

- Python 3.10
- NumPy 1.26
- PyTorch 2.6.0 + CUDA 12.4
- PyYAML 6.0
- matplotlib 3.10
- Pillow 11.3

安装依赖：

```bash
conda activate phf_env
pip install -r requirements.txt
```

如果只生成 GIF，不需要安装系统 ffmpeg。只有输出 `.mp4` 时才需要可用的 ffmpeg。

## 快速评估

仓库保留了一个代表性 checkpoint：

```bash
conda activate phf_env

python -m stage2.channel_evaluate \
  --config config_stage2_channel_probe.yaml \
  --scenario randomized \
  --policy checkpoint \
  --checkpoint checkpoints/stage2_channel_v3_gpu2_best.pt \
  --episodes 100 \
  --output-dir results/eval_quick
```

生成 GIF：

```bash
conda activate phf_env

python -m stage2.visualize \
  --config config_stage2_channel_probe.yaml \
  --scenario randomized \
  --policy checkpoint \
  --checkpoint checkpoints/stage2_channel_v3_gpu2_best.pt \
  --seed 0 \
  --output results/visualization/stage2_channel_v3_seed0.gif
```

## 服务器训练

4 GPU 训练当前主线场景：

```bash
cd ~/phf/p_g_weibu
conda activate phf_env

MONITOR_INTERVAL=60 bash stage2/launch_4gpu_channel_vector.sh \
  config_stage2_channel_probe.yaml \
  stage2_channel_v3
```

训练结束后评估并打包结果：

```bash
cd ~/phf/p_g_weibu
conda activate phf_env

bash stage2/evaluate_4gpu_channel.sh \
  config_stage2_channel_probe.yaml \
  stage2_channel_v3 \
  300

tar -czf stage2_channel_v3_artifacts.tar.gz \
  checkpoints_stage2_channel_v3 \
  logs_stage2_channel_v3 \
  server_runs_stage2_channel_v3 \
  results_stage2 \
  stage2_generated_configs \
  config_stage2_channel_probe.yaml
```

`hard_v1` 仅作为泛化压力测试使用，不是论文最小复现的必要部分。

## 测试

```bash
conda activate phf_env
python -m unittest discover -s tests
python -m unittest discover -s tests_stage2
```

## 复现结论

本工程实现了论文核心方法的二维简化版本：通过策略引导降低探索难度，再用强化学习策略做残差修正。在受控多障碍平面场景中可以稳定完成围捕，并提供可视化结果。

当前结果不应解释为“任意场景通用围捕策略”。要实现更强泛化，需要继续加入安全屏障、分阶段随机化、动态障碍/移动目标课程和更严格的多场景评估。
