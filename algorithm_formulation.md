# Algorithm Formulation

## 1. MDP / Dec-POMDP 建模

论文在第 III-A 节以 MDP 元组 `<S, A, T, R, gamma>` 描述 MARL，并在第 IV-C.2 节使用局部观测和双向通信。因此工程实现按 Dec-POMDP 组织：

- 全局状态 `s_t`：所有 pursuers 的位置、速度，target 的位置、速度，所有障碍物的位置、半径和安全 buffer。论文未给出 critic 的精确全局输入向量，属于工程假设。
- 局部观测 `o_t^i = S_t^i`：公式 (19)，由左邻居、右邻居、target、最近 obstacle 特征拼接。
- 联合动作 `a_t = [a_t^1, ..., a_t^N]`，其中 `a_t^i = u_i` 是二维加速度。来源：第 II-A 节公式 (1)、第 III-A 节。
- 转移 `T(s_{t+1}|s_t,a_t)`：由二阶积分器和速度/加速度限制决定。来源：第 II-A 节公式 (1)。
- 奖励 `R_i(s_t,a_t)`：GERD，公式 (15)-(18)。来源：第 IV-C.1 节。
- 折扣因子 `gamma = 0.985`。来源：第 V-A 节。

## 2. 智能体 agent 的定义

每个 pursuer 是一个 agent：

- 输入：本地观测 `S_t^i`。来源：第 IV-C.2 节公式 (19)。
- 输出：二维加速度动作 `u_i = [u_ix, u_iy]`。来源：第 II-A 节公式 (1)。
- 参数共享：所有 actor 网络结构相同并共享参数。来源：第 IV-C.3 节。
- 通信：只与左右邻居交换信息。来源：第 IV-C.2 节。

Target 不是学习 agent，在论文中是环境实体；其逃逸策略在实验中可为 PFP、GP、AP、MCP。来源：第 V-D 节公式 (20)-(23)。

## 3. 环境 environment 的定义

环境包含：

- `N` 个 pursuers，圆形实体。
- `M` 个静态 obstacles，圆形实体。
- 一个移动 target。
- 二维无界工作空间 `W subset R^2`。来源：第 II-A 节。

环境接口：

- `reset(seed) -> observations, state`
- `step(actions) -> observations, rewards, dones, info`
- `compute_observations()`
- `compute_rewards()`
- `check_encirclement()`
- `check_collisions()`

## 4. 状态 state

可实现的最小全局状态：

```text
pursuer_pos: shape [N, 2]
pursuer_vel: shape [N, 2]
target_pos: shape [2]
target_vel: shape [2]
obstacles: list of [x, y, radius, buffer]
finished_flags: shape [N]
step_count: int
```

来源：第 II-A 节实体定义和公式 (1)。critic 使用状态的具体拼接顺序缺失，最小实现将其作为工程假设。

## 5. 局部观测 observation

论文公式 (19)：

```text
S_i = concat(S_i_left, S_i_right, S_i_target, S_i_obstacle)
```

最小实现维度：

- 左邻居 5 维：`[alpha_ij, d_ij, rel_speed_ij, Q_ij, Q_ji]`
- 右邻居 5 维：同上
- 目标 6 维：`[d_iT, dot_d_iT, Q_iT, dot_Q_iT, omega, dot_omega]`
- 最近障碍物 8 维：`[x_io, y_io, d_io, d_jo, d_ko, r_o, v_ix, v_iy]`
- 合计 24 维。

来源：第 IV-C.2 节公式 (19) 和图 5。`dot_Q_iT`、`omega_dot` 的离散计算未明确，最小实现用当前几何量近似或置零，标为工程假设。

## 6. 动作 action

每个 agent 动作为二维加速度：

```text
u_i = [u_ix, u_iy]
```

约束：每个分量裁剪到 `[-u_max, u_max]`，速度每个分量裁剪到 `[-v_max, v_max]`。来源：第 II-A 节。

## 7. 奖励 reward

论文 GERD：

```text
if all pursuers finished:
    R_i = R_global
elif pursuer i finished:
    R_i = R_local
else:
    R_i = R_step
```

其中：

```text
R_f = exp(-k1 * norm(sum_i (p_T - p_i))) - 1
R_d = exp(-k2 * sum_i (d_iT - d_c)^2) - 1
R_step = w1 * R_f + w2 * R_d + w3 * R_l
```

默认：`R_l = -3`，`R_local = 5`，`R_global = 10`，`w1 = 0.3`，`w2 = 0.3`，`w3 = 0.4`。来源：第 IV-C.1 节公式 (15)-(18)、第 V-A 节。

注意：碰撞 penalty `R_l` 是否只在碰撞时取 -3、无碰撞时取 0，论文文字称 “penalty for collisions”，但公式没有显式指示非碰撞值。最小实现采用“碰撞为 -3，否则 0”，并写入 assumptions.md。

## 8. 状态转移 transition

对每个 pursuer：

```text
v_i[t+1] = clip(v_i[t] + clip(u_i, -u_max, u_max) * dt, -v_max, v_max)
p_i[t+1] = p_i[t] + v_i[t+1] * dt
```

对 target：由目标策略产生 `u_T` 或 `v_T`，再裁剪到论文约束。来源：第 II-A 节公式 (1)、第 V-D 节逃逸策略公式 (20)-(23)。

最小实现支持 static / greedy target policy；PFP、AP、MCP 为后续扩展。

## 9. 终止条件 done

最小实现采用：

- 成功：所有 pursuers 同时满足公式 (2)。
- 失败：任一 pursuer 与障碍物或队友碰撞。
- 截断：达到 `max_steps = 200`。

来源：第 II-A 节定义 2、第 V-D 节失败判定。

## 10. 多智能体交互机制

- 邻居定义：按 target-centered polar angle 排序，相邻者为左右邻居；方向按第 II-A 节 Definition 1 的左右邻居定义。工程上用极角循环排序实现。
- 双向通信：agent `i` 仅使用左邻居和右邻居信息，另加 target 和最近 obstacle。来源：第 IV-C.2 节。
- 奖励：`R_f`、`R_d` 由全局几何量计算，但发给各 agent 的 `R_i` 按 GERD 分段。来源：第 IV-C.1 节。

## 11. 集中式训练 / 分布式执行

论文采用 centralized training and decentralized execution：

- actor：每个 pursuer 基于局部观测执行，参数共享。来源：第 IV-C.3 节。
- critic：集中式 critic 评价整体状态。来源：第 IV-C.3 节。

最小实现先提供可运行的 APF guide policy 和 NumPy actor smoke model；完整 rMAPPO 更新接口保留，但不声称复现原论文训练曲线。

## 12. 策略网络、价值网络或评论家网络

论文表 II：

- Actor MLP layers：2
- Actor hidden units：128, 64
- Actor RNN hidden units：32
- Critic MLP layers：3
- Critic hidden units：256, 128, 64
- Critic RNN hidden units：32
- RNN data chunk length：10
- Updating batch size：100
- Base learning rate：0.0002
- Fine Tune learning rate：0.00015
- PPO update epoch：15

来源：第 IV-C.3 节、表 II。

最小实现说明：本地环境当前缺少 PyTorch，工程将保留 PyTorch 依赖建议，但 runnable smoke tests 使用 NumPy 实现 `LinearGaussianActor` 和 `LinearValueCritic`，只验证维度和循环。

## 13. 损失函数

- Policy loss：PPO clipped objective，公式 (6)。来源：第 III-A 节。
- Value loss：论文仅记为 `L^V(phi)`，未给出具体形式。最小实现采用 MSE value loss 作为工程假设。
- Advantage：Algorithm 1 第 9 行要求估计 `A_t`，论文提到 generalized advantage estimate，但未给出 `lambda`。最小实现训练 smoke 不做完整 PPO，只做 rollout 和简单 policy placeholder。

## 14. 训练更新过程

论文 Algorithm 1 可实现为：

1. episode 开始根据进度更新 regulators。
2. 前 `M^g` 步执行 APF guide policy，后续执行 learned policy。
3. 收集 `(o, a, r, next_o, done)`。
4. 计算 advantage。
5. 重复 `K` 次 PPO 更新 actor 和 critic。

最小工程实现：

- 完整实现环境、reward、regulators、APF guide policy、actor forward。
- 训练循环执行短 rollout，并记录总奖励。
- 不实现完整 PPO 梯度更新，标记为未完全复现。

## 15. 推理或测试过程

推理时每个 pursuer：

1. 从环境获得局部观测 `S_t^i`。
2. actor 输出加速度 `u_i`，或使用 APF guide policy 输出加速度。
3. 环境执行动作并返回下一个观测。
4. 当公式 (2) 成立、碰撞或步数达到上限时终止。

来源：第 IV-C.2/3 节、Algorithm 1。
