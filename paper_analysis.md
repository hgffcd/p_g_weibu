# Paper Analysis

## 1. 论文题目、研究对象和核心任务

- 题目：A Policy-Guided Reinforcement Learning Method for Encirclement Control in Multiobstacle Environment。
- 研究对象：二维无界空间中的多追捕者、多静态障碍物、移动目标围捕控制问题，论文称为 EMOCA，即 multiagent encirclement with multiobstacle collision avoidance。来源：摘要，第 II-A 节。
- 核心任务：多个 pursuers 在避免与障碍物、队友碰撞的同时，围绕移动 target 形成以目标为圆心、半径为捕获距离 `d_c` 的均匀环形编队。来源：第 II-A 节、定义 2、公式 (2)。

## 2. 论文要解决的问题

论文指出，EMOCA 难点在于同时平衡移动目标围捕和多障碍物避碰。纯 DRL 方法通常依赖碰撞负奖励；在障碍物多时，长时间负反馈会导致训练发散，且终端围捕附近的碰撞惩罚可能抵消成功围捕奖励。来源：第 I 节。

## 3. 论文提出的主要方法

论文提出 MRA-RLEC，即 multiregulator-assisted RL for encirclement control。核心组成如下：

- 使用 APF 人工势场作为 guide policy，配合 jump-start RL 缩小早期探索空间。来源：摘要、第 IV-A 节、公式 (7)-(10)。
- 使用 curriculum learning 的多个 regulator：JSRL regulator、angle regulator、distance regulator、obstacle avoidance regulator。来源：第 IV-B 节、图 4、公式 (11)-(14)。
- 使用 GERD，全局围捕奖励分解方法，为部分围捕成功提供局部奖励，缓解稀疏奖励。来源：第 IV-C.1 节、公式 (15)-(18)。
- 使用 bidirectional communication protocol，每个 pursuer 只和左右邻居通信。来源：第 IV-C.2 节、公式 (19)。
- 基础训练算法采用带 recurrent layers 的 rMAPPO；论文也验证该框架可叠加到 IPPO、MAPPO、HAPPO、MADDPG、MATD3 等 MARL 算法。来源：第 III-A 节、第 V-B 节。

## 4. 整体算法流程

论文算法见 Algorithm 1：

1. 初始化 actor 参数 `theta` 和 critic 参数 `phi`。来源：Algorithm 1，第 1 行。
2. 每个 episode 计算训练进度 `E_t / E`，选择相关 regulators。来源：Algorithm 1，第 2-4 行。
3. 用公式 (12)、(13) 更新围捕条件偏差 `Delta_alpha`、`Delta_d`，用公式 (14) 更新障碍物半径。来源：Algorithm 1，第 3 行。
4. 用公式 (11) 计算 JSRL regulator，并得到 guide steps `M^g`。来源：Algorithm 1，第 4 行。
5. episode 内前 `M^g` 步执行 guide policy `pi^g`，其余步执行 learned policy `pi^theta`，收集轨迹。来源：Algorithm 1，第 5-8 行。
6. 从轨迹估计 advantage，并按 PPO clipped objective 更新 actor，按 value loss 更新 critic。来源：Algorithm 1，第 9-14 行、公式 (6)。

## 5. 关键公式及其含义

- 公式 (1)：追捕者二阶积分器动力学，`dot p_i = v_i`，`dot v_i = u_i`。含义：动作为二维加速度，积分得到速度和位置。来源：第 II-A 节。
- 公式 (2)：围捕条件，要求与左右邻居夹角误差小于 `Delta_alpha`，且与目标距离误差小于 `Delta_d`。来源：第 II-A 节、定义 2。
- 公式 (3)：MARL 目标，最大化折扣累计奖励期望。来源：第 III-A 节。
- 公式 (5)、(6)：PPO likelihood ratio 和 clipped policy loss。来源：第 III-A 节。
- 公式 (7)：追捕者之间的斥力，仅当距离 `d_ij <= L` 时生效，`L = 2 d_c sin(alpha_exp / 2)`。来源：第 IV-A 节。
- 公式 (8)：追捕者和障碍物之间的斥力，仅当距离 `d_ik <= L_s` 时生效。来源：第 IV-A 节。
- 公式 (9)：追捕者与目标之间的吸引力，与 `d_iT - d_c` 成比例。来源：第 IV-A 节。
- 公式 (10)：guide policy 的总加速度，包含 pursuer-pursuer 斥力、pursuer-obstacle 斥力、target attraction 和阻尼项。来源：第 IV-A 节。
- 公式 (11)：S-shaped JSRL regulator，用于平滑减少 guide policy 步数。来源：第 IV-A 节。
- 公式 (12)、(13)：angle / distance regulator，线性降低终端条件允许偏差。来源：第 IV-B 节。
- 公式 (14)：obstacle avoidance regulator，让障碍物半径从 0 逐步增长到真实半径。来源：第 IV-B 节。
- 公式 (15)：GERD 奖励分段定义，全部完成给 `R_global`，单个 pursuer 完成给 `R_local`，否则给 `R_step`。来源：第 IV-C.1 节。
- 公式 (16)、(17)：formation reward 和 distance reward。来源：第 IV-C.1 节。
- 公式 (18)：step reward 加权和。来源：第 IV-C.1 节。
- 公式 (19)：actor 输入状态向量，由左邻居、右邻居、目标、最近障碍物状态拼接而成。来源：第 IV-C.2 节。

## 6. 输入数据和输出结果

- 输入：环境参数 `N, M, d_c, Delta_alpha, Delta_d, r_i, delta_i, obstacle states, p_T, v_T`，训练参数 `E, M, lr, K`，网络参数。来源：第 II-A 节、Algorithm 1、表 II。
- 输出：训练后的 actor network `theta` 和 critic network `phi`。来源：Algorithm 1。
- 实验输出：围捕轨迹、训练 reward、平均 step reward、encirclement time、danger rate、success rate、距离误差和角度误差。来源：第 V-B 至 V-F 节、表 IV、表 V、图 6-14。

## 7. 状态空间、动作空间、观测空间

- 全局状态：论文在第 III-A 节用 MDP 元组 `S, A, T, R, gamma` 描述 MARL，但未给出完整全局状态向量的实现格式；集中式 critic 使用整体状态。缺失信息：critic 输入的精确拼接顺序和维度未完整列出。来源：第 III-A 节、第 IV-C.3 节。
- 局部观测：每个 pursuer 的输入 `S_t^i = [S_t^{i,j}, S_t^{i,k}, S_t^{i,T}, S_t^{i,o}]`。来源：第 IV-C.2 节、公式 (19)。
- 邻居观测：`S_t^{i,j} = [alpha_ij, d_ij, v_ij, Q_ij, Q_ji]^T`，右邻居同构。来源：第 IV-C.2 节、图 5。
- 目标观测：`S_t^{i,T} = [d_iT, dot d_iT, Q_iT, dot Q_iT, omega, dot omega]^T`。来源：第 IV-C.2 节。
- 最近障碍物观测：`S_t^{i,o} = [x_io, y_io, d_io, d_jo, d_ko, r_o, v_ix, v_iy]^T`。来源：第 IV-C.2 节。
- 动作：每个 pursuer 输出二维加速度 `u_i = [u_ix, u_iy]`。来源：第 II-A 节、公式 (1)。

## 8. 奖励函数或目标函数

GERD 奖励：

- 若全部 pursuers finished：`R_i = R_global`。
- 若 pursuer `i` finished：`R_i = R_local`。
- 否则：`R_i = R_step`。
- `R_step = w1 R_f + w2 R_d + w3 R_l`，其中 `R_l = -3`，`R_local = 5`，`R_global = 10`；仿真设置 `w1 = 0.3, w2 = 0.3, w3 = 0.4`。来源：第 IV-C.1 节、公式 (15)-(18)、第 V-A 节。

## 9. 约束条件

- 动力学：二阶积分器，公式 (1)。来源：第 II-A 节。
- 加速度约束：每个加速度分量有 `u_max` 限制；target 同样受限。来源：第 II-A 节。
- 速度约束：pursuer 最大速度 `v_max`，target 最大速度 `v_Tmax`，并设置 `u_max = u_Tmax, v_max > v_Tmax`。来源：第 II-A 节。
- 围捕条件：公式 (2)。来源：第 II-A 节。
- 碰撞约束或惩罚条件：当 `||p_i - p_j|| < r_i + r_j + k_delta (delta_i + delta_j)` 时给予惩罚；障碍物也需要安全缓冲。来源：第 II-A 节。

## 10. 训练流程或优化流程

- Base 阶段：使用 JSRL regulator、angle regulator、distance regulator、obstacle avoidance regulator，从简单任务逐步恢复到原始难度。来源：第 IV-B 节。
- Fine Tune 阶段：引入不同 target escape policies，线性提高目标速度，增大碰撞惩罚，降低学习率；不使用 guide policy。来源：第 IV-B 节、第 V-B 节。
- 网络更新：rMAPPO，actor 使用 PPO clipped loss，critic 使用 value loss。来源：第 III-A 节、第 IV-C.3 节、Algorithm 1。

## 11. 仿真环境或实验环境

- 数值仿真：二维环境，默认 pursuer 数 `N = 5`，实验也测试 `N = 3,4,5,6`。来源：第 V 节。
- Gazebo + ROS2 半物理仿真：自设计全向车模型，actor 输出加速度，之后通过运动学转换成轮速并用 ROS2 发布控制命令。来源：第 II-B 节、第 V-F 节。
- 表 I 车辆参数：质量 0.8 kg，半径 0.15 m，pursuer 最大速度 0.5 m/s，evader 最大速度 0.25 m/s，最大加速度均 0.5 m/s^2，惯量 `I_xx = I_yy = 0.006 kg*m^2`。来源：第 II-B 节、表 I。
- 第 V-A 节仿真参数：`v_max = 0.5 m/s`，Base 中 `v_Tmax = 0.15 m/s`，Fine Tune 中 `v_Tmax = 0.25 m/s`，`u_max = 0.5 m/s^2`，`d_c = 1 m`，`Delta_alpha = 0.3 rad`，`Delta_d = 0.1 m`，`k_r = 4.5`，`k_o = 4.0`，`k_a = 2.0`，`k_b = 1.5`，`L_s = 0.3 m`，episode 最大步数 200，`dt = 0.1 s`。
- 训练规模：Base `7.68e6` steps，Fine Tune `5.12e6` steps。来源：第 V-A 节。

## 12. 评价指标

- average step reward、encirclement time、danger rate、successful rate。来源：第 V-D 节、表 IV。
- danger rate：安全缓冲判定下实体过近的频率。来源：第 V-D 节。
- success：不满足围捕条件或发生队友/障碍物碰撞则失败。来源：第 V-D 节。
- 距离误差 `e_d = (1/N) sum_i |d_iT - d_c|`；角度误差 `e_alpha = (1/N) sum_i |alpha_{i,i+1} - alpha_exp|`。来源：第 V-F 节、图 14。
- 碰撞距离指标：`D_o` 为 pursuer 到障碍物最小距离扣除障碍半径，`D_p` 为 pursuer 间最小距离扣除 pursuer 半径。来源：第 V-F 节、图 13。

## 13. 需要实现的核心模块

- 二维 EMOCA 环境：reset、step、二阶积分器、速度/加速度裁剪、目标运动、碰撞检测、围捕判定。来源：第 II-A 节、公式 (1)、公式 (2)。
- 邻居排序和左右邻居选择。来源：第 II-A 节、Definition 1。
- 局部观测构造和双向通信。来源：第 IV-C.2 节、公式 (19)。
- APF guide policy。来源：第 IV-A 节、公式 (7)-(10)。
- curriculum regulators。来源：第 IV-A/B 节、公式 (11)-(14)。
- GERD reward。来源：第 IV-C.1 节、公式 (15)-(18)。
- actor/critic 网络或最小策略模块。来源：第 IV-C.3 节、表 II。
- rMAPPO 训练循环或最小 smoke training loop。来源：第 III-A 节、Algorithm 1。

## 14. 论文中没有明确说明、需要工程假设的部分

- 初始 pursuer 直线位置的具体坐标、目标随机圆区域的圆心和半径未给出。来源：第 II-A 节只描述初始化方式。
- 障碍物数量、具体坐标、半径分布和安全 buffer 参数 `delta`、`k_delta` 未完整给出。来源：第 II-A 节。
- target 在 Base 训练中的运动策略未完整公式化；Fine Tune 使用 PFP，但具体训练任务采样细节未完整给出。来源：第 V-B 节。
- critic 全局状态的精确向量格式未给出。来源：第 IV-C.3 节。
- value loss `L^V(phi)`、GAE 参数、optimizer、标准化、batch 组织、多线程采样细节未完整给出。来源：第 III-A 节、Algorithm 1、表 II。
- 网络初始化、Gaussian action distribution 的标准差处理、动作裁剪策略未详细说明。来源：第 IV-C.3 节、图 3。
- 表 IV/V 的具体数值不影响最小复现，本文档仅引用其指标定义；完整实验复现需要原始环境配置与训练脚本。来源：第 V-D/E 节。
