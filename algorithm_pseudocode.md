# Algorithm Pseudocode

## Pseudocode 1: Environment Reset

对应论文：第 II-A 节初始化描述；缺失具体坐标，以下坐标采样为工程假设。

```python
def reset(seed):
    rng = Random(seed)

    # 第 II-A 节：pursuers initially aligned in a straight line.
    pursuer_pos = make_line_positions(num_pursuers, spacing=0.8)  # 工程假设
    pursuer_vel = zeros([num_pursuers, 2])                         # 第 II-A 节

    # 第 II-A 节：target positioned randomly in a circular area ahead.
    target_pos = sample_point_in_front_circle(rng)                 # 工程假设
    target_vel = zeros([2])                                        # 第 II-A 节

    # 第 II-A 节：M static circular obstacles.
    obstacles = load_or_sample_obstacles(rng)                      # 工程假设

    finished = [False] * num_pursuers
    step_count = 0
    return compute_observations(), get_global_state()
```

## Pseudocode 2: Neighbor Selection

对应论文：第 II-A 节 Definition 1。

```python
def get_left_right_neighbors(pursuer_pos, target_pos):
    rel = pursuer_pos - target_pos
    angles = atan2(rel[:, 1], rel[:, 0])
    order = argsort(angles)

    neighbors = {}
    for rank, i in enumerate(order):
        left = order[(rank - 1) % num_pursuers]
        right = order[(rank + 1) % num_pursuers]
        neighbors[i] = (left, right)

    return neighbors
```

说明：论文用向量叉乘定义左右邻居。工程实现用 target-centered 极角循环排序，与“夹角中无其他 pursuer 则为邻居”的定义等价。

## Pseudocode 3: Observation Construction

对应论文：第 IV-C.2 节、公式 (19)、图 5。

```python
def compute_observation(i):
    left, right = neighbors[i]
    closest_obstacle = argmin_obstacle_distance(i)

    # 第 IV-C.2 节：S_t^{i,j} = [alpha_ij, d_ij, v_ij, Q_ij, Q_ji]^T
    left_features = neighbor_features(i, left)
    right_features = neighbor_features(i, right)

    # 第 IV-C.2 节：S_t^{i,T} = [d_iT, dot_d_iT, Q_iT, dot_Q_iT, omega, dot_omega]^T
    target_features = target_features(i)

    # 第 IV-C.2 节：S_t^{i,o} = [x_io, y_io, d_io, d_jo, d_ko, r_o, v_ix, v_iy]^T
    obstacle_features = obstacle_features(i, left, right, closest_obstacle)

    # 公式 (19)
    return concat(left_features, right_features, target_features, obstacle_features)
```

`dot_Q_iT` 和 `dot_omega` 的离散估计未明确；最小实现置零或用上一帧差分，标为工程假设。

## Pseudocode 4: Encirclement Check

对应论文：第 II-A 节 Definition 2、公式 (2)。

```python
def check_finished_each_agent():
    neighbors = get_left_right_neighbors()
    alpha_exp = 2 * pi / num_pursuers
    finished = []

    for i in pursuers:
        left, right = neighbors[i]
        distance_ok = abs(distance(p_i, p_target) - capture_distance) < delta_d

        # 公式 (2)：对任意邻居 j in N(i)，|alpha_ij - alpha_exp| < Delta_alpha
        left_angle_ok = abs(included_angle(i, left, target) - alpha_exp) < delta_alpha
        right_angle_ok = abs(included_angle(i, right, target) - alpha_exp) < delta_alpha

        finished.append(distance_ok and left_angle_ok and right_angle_ok)

    return finished
```

## Pseudocode 5: Collision Check

对应论文：第 II-A 节安全 buffer 惩罚条件；第 V-D 节失败判定。

```python
def check_collisions():
    for each pair of pursuers (i, j):
        if norm(p_i - p_j) < r_i + r_j + k_delta * (delta_i + delta_j):
            return True

    for pursuer i and obstacle o:
        if norm(p_i - p_o) < r_i + r_o + k_delta * (delta_i + delta_o):
            return True

    return False
```

## Pseudocode 6: APF Guide Policy

对应论文：第 IV-A 节、公式 (7)-(10)。

```python
def guide_action_for_agent(i):
    alpha_exp = 2 * pi / num_pursuers
    L = 2 * capture_distance * sin(alpha_exp / 2)

    total = zeros(2)

    # 公式 (7)：pursuer-pursuer repelling force
    for j in pursuers where j != i:
        d_ij = norm(p_i - p_j)
        if d_ij <= L:
            total += k_r * (L - d_ij) / d_ij * (p_i - p_j)

    # 公式 (8)：pursuer-obstacle repelling force
    for obstacle k in obstacles:
        d_ik = norm(p_i - p_obstacle_k)
        if d_ik <= L_s:
            total += k_o * (L_s - d_ik) / d_ik * (p_i - p_obstacle_k)

    # 公式 (9)：target attraction force
    d_iT = norm(p_i - p_target)
    total += k_a * (d_iT - capture_distance) / d_iT * (p_target - p_i)

    # 公式 (10)：damping term
    total -= k_b * v_i

    return clip(total, -u_max, u_max)
```

## Pseudocode 7: Curriculum Regulators

对应论文：第 IV-A/B 节、公式 (11)-(14)。

```python
def jsrl_regulator(x, k_s):
    # 公式 (11)
    X = 2 * x - 1
    epsilon = 1 + tanh(-k_s)
    return 0.5 * (-tanh(k_s * X) - epsilon * X**3 + 1)

def guide_steps(E_t, E, M):
    # 第 IV-A 节：E_g = c_g E, M^g = M * rho_0 * f(E_t / E_g)
    E_g = c_g * E
    if E_t >= E_g:
        return 0
    ratio = E_t / E_g
    return floor(M * rho_0 * jsrl_regulator(ratio, k_s))

def angle_distance_bias(E_t, E):
    # 公式 (12)(13)：E_e = c_e E
    E_e = c_e * E
    ratio = min(E_t / E_e, 1.0)
    delta_alpha = delta_alpha_start - (delta_alpha_start - delta_alpha_expected) * ratio
    delta_d = delta_d_start - (delta_d_start - delta_d_expected) * ratio
    return delta_alpha, delta_d

def obstacle_radius(E_t, E, original_radius):
    # 公式 (14)：E_o^lo = c_lo E, E_o^hi = c_hi E
    lo = c_lo * E
    hi = c_hi * E
    if E_t < lo:
        return 0
    if E_t > hi:
        return original_radius
    return original_radius * (E_t - lo) / (hi - lo)
```

## Pseudocode 8: GERD Reward

对应论文：第 IV-C.1 节、公式 (15)-(18)。

```python
def compute_rewards():
    finished = check_finished_each_agent()
    collision = check_collisions()

    # 公式 (16)
    aggregate = sum(target_pos - pursuer_pos[i] for i in pursuers)
    R_f = exp(-k1 * norm(aggregate)) - 1

    # 公式 (17)
    dist_error_sum = sum((distance(pursuer_pos[i], target_pos) - capture_distance)**2 for i in pursuers)
    R_d = exp(-k2 * dist_error_sum) - 1

    # 第 IV-C.1 节：collision penalty R_l = -3.
    # 工程假设：无碰撞时 R_l = 0。
    R_l = collision_penalty if collision else 0

    # 公式 (18)
    R_step = w1 * R_f + w2 * R_d + w3 * R_l

    rewards = []
    for i in pursuers:
        # 公式 (15)
        if all(finished):
            rewards.append(R_global)
        elif finished[i]:
            rewards.append(R_local)
        else:
            rewards.append(R_step)
    return rewards
```

## Pseudocode 9: Environment Step

对应论文：第 II-A 节公式 (1)、Algorithm 1 第 5-8 行。

```python
def step(actions):
    actions = clip(actions, -u_max, u_max)

    # 公式 (1)
    pursuer_vel = clip(pursuer_vel + actions * dt, -v_max, v_max)
    pursuer_pos = pursuer_pos + pursuer_vel * dt

    target_action = target_policy()
    target_vel = clip(target_vel + target_action * dt, -v_target_max, v_target_max)
    target_pos = target_pos + target_vel * dt

    rewards = compute_rewards()
    finished = check_finished_each_agent()
    collision = check_collisions()
    step_count += 1

    done = all(finished) or collision or step_count >= max_steps
    info = {"success": all(finished), "collision": collision, "finished": finished}

    return compute_observations(), rewards, done, info
```

## Pseudocode 10: Training Loop

对应论文：Algorithm 1、公式 (6)。最小实现只做 smoke rollout；完整 PPO 更新标为未完全复现。

```python
def train(config):
    actor = Actor(obs_dim, action_dim)      # 第 IV-C.3 节、表 II
    critic = Critic(global_state_dim)       # 第 IV-C.3 节、表 II
    guide = APFGuidePolicy(config)          # 第 IV-A 节

    for E_t in range(1, total_episodes + 1):
        delta_alpha, delta_d = angle_distance_bias(E_t, total_episodes)
        update_obstacle_radii(E_t, total_episodes)
        M_g = guide_steps(E_t, total_episodes, max_steps)

        obs, state = env.reset()
        trajectory = []

        for step in range(max_steps):
            if step < M_g:
                actions = guide.act(env.state)       # Algorithm 1 第 6 行
            else:
                actions = actor.act(obs)             # Algorithm 1 第 6 行

            next_obs, rewards, done, info = env.step(actions)
            trajectory.append((obs, actions, rewards, next_obs, done))
            obs = next_obs
            if done:
                break

        # Algorithm 1 第 9-13 行；完整 PPO 为后续工作。
        # advantages = estimate_gae(trajectory)
        # for k in range(K):
        #     policy_loss = ppo_clipped_loss(actor, trajectory, advantages)  # 公式 (6)
        #     value_loss = mse(critic(state), returns)                       # 工程假设
        #     optimizer.step()
```

## Pseudocode 11: Evaluation

对应论文：第 V-D/F 节评价指标。

```python
def evaluate(policy, episodes):
    for episode in range(episodes):
        obs, state = env.reset()
        total_reward = 0
        for t in range(max_steps):
            actions = policy.act(obs)
            obs, rewards, done, info = env.step(actions)
            total_reward += mean(rewards)
            if done:
                break

        record(
            success=info["success"],                  # 第 V-D 节
            collision=info["collision"],              # 第 V-D 节
            encirclement_time=t * dt if success else None,
            distance_error=mean(abs(d_iT - d_c)),     # 第 V-F 节图 14
            angle_error=mean(abs(alpha_i_next - alpha_exp)),
        )
```
