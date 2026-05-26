# 07: Reinforcement Learning

Deep reinforcement learning — from value-based methods (DQN) to policy gradient (PPO) — all implemented from scratch in PyTorch on OpenAI Gym environments.

## Notebooks

| Notebook | Algorithm | Key Idea |
|----------|-----------|---------|
| `01-DQN.ipynb` | Deep Q-Network | Q-learning + neural network + replay buffer + target network |
| `02-DDQN.ipynb` | Double DQN | Decouple action selection from value estimation to reduce overestimation |
| `03-PPO.ipynb` | Proximal Policy Optimization | On-policy actor-critic with clipped surrogate objective |

## Algorithm Overview

### Value-Based Methods (DQN Family)

```
Environment → State s
    │
    ▼
Q-Network  Q(s, a; θ)         ← outputs Q-value for each action
    │
    ▼
Select action: ε-greedy
    │
    ▼
Store (s, a, r, s') in Replay Buffer
    │
    ▼
Sample mini-batch → Bellman update
    TD target: y = r + γ · max_a Q(s', a; θ⁻)   ← target network
    Loss: MSE(Q(s,a;θ), y)
```

### Policy Gradient (PPO)

```
Actor-Critic Network
    ├── Actor:  π(a|s; θ)  → action probability distribution
    └── Critic: V(s; θ)    → state value estimate

Rollout → compute GAE advantage Â
    │
    ▼
K epochs of mini-batch updates:
    Clipped surrogate:  L_CLIP = E[min(r·Â, clip(r, 1-ε, 1+ε)·Â)]
    Value loss:         L_VF   = MSE(V(s), V_target)
    Entropy bonus:      L_ENT  = H[π(·|s)]    (encourages exploration)
    Total loss: L_CLIP − c₁·L_VF + c₂·L_ENT
```

## DQN vs DDQN vs PPO

| | DQN | DDQN | PPO |
|--|-----|------|-----|
| **Type** | Value-based | Value-based | Policy gradient |
| **On/Off policy** | Off-policy | Off-policy | On-policy |
| **Memory** | Replay buffer | Replay buffer | Rollout buffer (cleared each iteration) |
| **Key fix** | Target network | Separate action selection & evaluation | Clipped objective prevents large policy updates |
| **Environment** | Discrete actions | Discrete actions | Discrete or continuous actions |

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Replay buffer** | Store past transitions; break temporal correlation |
| **Target network** | Copy of Q-network updated every N steps; stabilises training |
| **ε-greedy** | Explore with probability ε, exploit otherwise |
| **GAE (λ)** | Generalised Advantage Estimation — balances bias/variance in advantage |
| **Clipped surrogate** | Prevents PPO from taking large policy update steps |
| **Entropy bonus** | Keeps policy from collapsing to deterministic too early |

## References

| Paper | Link |
|-------|------|
| DQN — Human-level control through deep reinforcement learning | [Nature 2015](https://www.nature.com/articles/nature14236) |
| DDQN — Deep Reinforcement Learning with Double Q-learning | [arxiv](https://arxiv.org/abs/1509.06461) |
| PPO — Proximal Policy Optimization Algorithms | [arxiv](https://arxiv.org/abs/1707.06347) |
| GAE — High-Dimensional Continuous Control Using GAE | [arxiv](https://arxiv.org/abs/1506.02438) |
