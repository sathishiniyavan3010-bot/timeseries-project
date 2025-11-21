

import os
import math
import random
from collections import deque, namedtuple
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------- Synthetic market generator ---------------------------

def generate_correlated_returns(n_assets: int, n_steps: int, seed: int = 42,
                                base_vol=0.01, vol_trend=0.00001, base_corr=0.2) -> np.ndarray:
    """Generate n_steps of daily returns for n_assets with time-varying vol and correlation.
    Returns shape: (n_steps, n_assets)
    """
    np.random.seed(seed)
    t = np.arange(n_steps)

    # volatility per asset varies slowly over time
    vols = base_vol * (1.0 + 0.5 * np.sin(2 * np.pi * t / 250)[:, None])  # seasonal
    vols += vol_trend * t[:, None]  # small trending volatility

    # base correlation matrix (toeplitz-like)
    A = np.eye(n_assets) * 1.0
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            corr = base_corr * (1.0 - 0.5 * abs(i-j) / n_assets)
            A[i, j] = corr
            A[j, i] = corr

    # ensure positive-definite
    eigs = np.linalg.eigvals(A)
    if np.any(eigs <= 0):
        A += np.eye(n_assets) * 0.01

    returns = np.zeros((n_steps, n_assets))
    for i in range(n_steps):
        # small time-varying correlation perturbation
        corr_noise = 0.05 * np.sin(2 * np.pi * i / 125)
        C = A + corr_noise * (np.ones_like(A) - np.eye(n_assets))
        # construct covariance
        cov = (vols[i] @ vols[i].T) * C  # outer product of vols * correlation
        # sample returns
        r = np.random.multivariate_normal(np.zeros(n_assets), cov)
        returns[i] = r
    return returns


# --------------------------- Portfolio environment ---------------------------
class PortfolioEnv:
    """A minimal portfolio environment.

    State: last `lookback` returns (flattened) + previous weights
    Action: continuous vector of raw scores -> converted to weights via softmax
    Reward: portfolio return for the step minus risk parity penalty and transaction cost
    """
    def __init__(self, returns: np.ndarray, lookback: int = 20, tcost: float = 1e-4, rp_penalty: float = 10.0):
        self.returns = returns  # shape (T, N)
        self.T, self.N = returns.shape
        self.lookback = lookback
        self.tcost = tcost
        self.rp_penalty = rp_penalty

        self.reset()

    def reset(self, start_idx: int = None):
        if start_idx is None:
            self.idx = self.lookback
        else:
            self.idx = start_idx
        self.weights = np.ones(self.N) / self.N
        self.done = False
        return self._get_state()

    def step(self, action_raw: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        # action_raw: (N,) unnormalized scores
        weights = self._raw_to_weights(action_raw)

        # compute portfolio return between t and t+1: using returns at time idx (treated as one-period returns)
        r_t = self.returns[self.idx]  # shape (N,)
        portfolio_return = np.dot(weights, r_t)

        # transaction cost: proportional to L1 change in weights
        tc = self.tcost * np.sum(np.abs(weights - self.weights))

        # risk parity penalty: compute marginal risk contributions and penalize deviations from equal
        # compute covariance using recent lookback window
        cov = np.cov(self.returns[self.idx - self.lookback:self.idx].T)
        port_vol = math.sqrt(max(1e-12, weights @ cov @ weights))
        # marginal risk contribution: w_i * (Sigma w)_i / portfolio_vol
        mrc = weights * (cov @ weights)
        if port_vol > 0:
            mrc = mrc / (port_vol)
        else:
            mrc = np.zeros_like(mrc)
        avg_mrc = np.mean(mrc)
        rp_error = np.sum((mrc - avg_mrc)**2)

        # reward: portfolio return minus penalties
        reward = portfolio_return - tc - self.rp_penalty * rp_error

        # advance
        self.idx += 1
        if self.idx >= self.T:
            self.done = True

        self.weights = weights.copy()
        return self._get_state(), float(reward), self.done, {"portfolio_return": portfolio_return,
                                                              "tc": tc,
                                                              "rp_error": rp_error}

    def _get_state(self) -> np.ndarray:
        # returns of last lookback periods (lookback x N) flattened + current weights
        past = self.returns[self.idx - self.lookback:self.idx].flatten()
        state = np.concatenate([past, self.weights])
        return state.astype(np.float32)

    def _raw_to_weights(self, raw: np.ndarray) -> np.ndarray:
        # convert to positive weights summing to 1 using softmax (numeric stabilized)
        ex = np.exp(raw - np.max(raw))
        w = ex / (np.sum(ex) + 1e-12)
        return w


# --------------------------- Utilities & metrics ---------------------------

def cumulative_returns(returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1 + returns) - 1


def sharpe_ratio(returns: np.ndarray, freq=252.0) -> float:
    # assume returns are raw period returns
    if returns.size == 0:
        return 0.0
    mean = np.mean(returns) * freq
    std = np.std(returns) * math.sqrt(freq)
    if std == 0:
        return 0.0
    return mean / std


def max_drawdown(returns: np.ndarray) -> float:
    cr = cumulative_returns(returns)
    peak = np.maximum.accumulate(cr)
    dd = (cr - peak) / (peak + 1e-12)
    return float(np.min(dd))


# --------------------------- Simple DDPG agent (PyTorch) ---------------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, s):
        return self.net(s)  # raw scores; environment will softmax


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x).squeeze(-1)


class DDPGAgent:
    def __init__(self, state_dim, action_dim, device='cpu', actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, tau=1e-3):
        self.device = device
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.a_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.c_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau

    def select_action(self, state: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw = self.actor(s).cpu().numpy().squeeze(0)
        raw = raw + noise_scale * np.random.randn(*raw.shape)
        return raw

    def update(self, batch: Transition):
        s = torch.tensor(np.stack(batch.state)).float().to(self.device)
        a = torch.tensor(np.stack(batch.action)).float().to(self.device)
        r = torch.tensor(np.stack(batch.reward)).float().to(self.device)
        ns = torch.tensor(np.stack(batch.next_state)).float().to(self.device)
        d = torch.tensor(np.stack(batch.done)).float().to(self.device)

        # Critic loss
        with torch.no_grad():
            next_raw = self.actor_target(ns)
            # next action will be transformed by env softmax later; approximate by softmax here
            next_a = torch.softmax(next_raw, dim=-1)
            q_next = self.critic_target(ns, next_a)
            y = r + self.gamma * (1 - d) * q_next

        q = self.critic(s, a)
        critic_loss = nn.MSELoss()(q, y)

        self.c_optimizer.zero_grad()
        critic_loss.backward()
        self.c_optimizer.step()

        # Actor loss: maximize Q (or minimize -Q)
        raw = self.actor(s)
        a_pred = torch.softmax(raw, dim=-1)
        actor_loss = -self.critic(s, a_pred).mean()

        self.a_optimizer.zero_grad()
        actor_loss.backward()
        self.a_optimizer.step()

        # soft update targets
        for param, target in zip(self.actor.parameters(), self.actor_target.parameters()):
            target.data.copy_(self.tau * param.data + (1.0 - self.tau) * target.data)
        for param, target in zip(self.critic.parameters(), self.critic_target.parameters()):
            target.data.copy_(self.tau * param.data + (1.0 - self.tau) * target.data)

        return float(critic_loss.item()), float(actor_loss.item())


# --------------------------- Training loop ---------------------------

def train_ddpg(env: PortfolioEnv, agent: DDPGAgent, buffer: ReplayBuffer,
               epochs: int = 50, steps_per_epoch: int = 250, batch_size: int = 256,
               warmup: int = 1000, noise_scale: float = 0.1):
    state_dim = env.lookback * env.N + env.N
    action_dim = env.N

    total_steps = 0
    stats = {'episode_returns': [], 'episode_sharpe': [], 'rp_errors': []}

    for ep in range(epochs):
        state = env.reset()
        ep_returns = []
        done = False
        step = 0
        while not done and step < steps_per_epoch:
            # select action
            raw = agent.select_action(state, noise_scale=noise_scale)
            next_state, reward, done, info = env.step(raw)
            # action passed to buffer should be actual weights (post-softmax)
            action_weights = env._raw_to_weights(raw)
            buffer.push(state, action_weights.astype(np.float32), np.array([reward], dtype=np.float32), next_state, np.array([done], dtype=np.float32))

            state = next_state
            ep_returns.append(info['portfolio_return'])
            total_steps += 1
            step += 1

            # update
            if len(buffer) > batch_size and total_steps > warmup:
                batch = buffer.sample(batch_size)
                critic_loss, actor_loss = agent.update(batch)

        # end episode stats
        ep_returns = np.array(ep_returns)
        stats['episode_returns'].append(ep_returns.sum())
        stats['episode_sharpe'].append(sharpe_ratio(ep_returns))
        stats['rp_errors'].append(info['rp_error'])

        print(f'Epoch {ep+1}/{epochs} — EpReturn: {stats["episode_returns"][-1]:.4f} Sharpe: {stats["episode_sharpe"][-1]:.4f} RP_err: {stats["rp_errors"][-1]:.6f}')

    return stats


# --------------------------- Evaluation helpers ---------------------------

def evaluate_policy(env: PortfolioEnv, agent: DDPGAgent, start_idx=1000, horizon=500):
    env.reset(start_idx=start_idx)
    returns = []
    weights_history = []
    for _ in range(horizon):
        s = env._get_state()
        raw = agent.select_action(s, noise_scale=0.0)
        w = env._raw_to_weights(raw)
        _, reward, done, info = env.step(raw)
        returns.append(info['portfolio_return'])
        weights_history.append(w)
        if done:
            break
    returns = np.array(returns)
    weights_history = np.array(weights_history)
    return returns, weights_history


# --------------------------- Baselines ---------------------------

def equal_weight_baseline(returns: np.ndarray):
    w = np.ones(returns.shape[1]) / returns.shape[1]
    port_rets = returns @ w
    return port_rets


def mean_variance_offline(returns: np.ndarray, risk_aversion: float = 1.0):
    # simple mean-variance solution: maximize mu^T w - (risk_aversion/2) w^T Sigma w s.t. sum w=1, w>=0
    mu = returns.mean(axis=0)
    Sigma = np.cov(returns.T)
    # quadratic programming solver not included — use a crude heuristic: w proportional to mu / diag(Sigma)
    diag = np.diag(Sigma)
    score = mu / (diag + 1e-8)
    score = np.clip(score, 0.0, None)
    if score.sum() == 0:
        w = np.ones_like(score) / len(score)
    else:
        w = score / score.sum()
    port_rets = returns @ w
    return port_rets, w


# --------------------------- Example run ---------------------------
if __name__ == '__main__':
    # generate market
    N = 6
    T = 5000
    rets = generate_correlated_returns(N, T, seed=123, base_vol=0.003, base_corr=0.3)

    lookback = 30
    env = PortfolioEnv(rets, lookback=lookback, tcost=1e-4, rp_penalty=50.0)

    state_dim = lookback * N + N
    action_dim = N

    agent = DDPGAgent(state_dim, action_dim, device='cpu')
    buffer = ReplayBuffer(100000)

    stats = train_ddpg(env, agent, buffer, epochs=30, steps_per_epoch=200, batch_size=256, warmup=2000, noise_scale=0.2)

    # Evaluate agent
    eval_rets, weights = evaluate_policy(env, agent, start_idx=3000, horizon=1000)
    print('Agent cumulative return:', (np.prod(1+eval_rets)-1))
    print('Agent Sharpe:', sharpe_ratio(eval_rets))
    print('Agent Max drawdown:', max_drawdown(eval_rets))

    # Baseline comparisons
    test_returns = rets[3000:3000+len(eval_rets)]
    ew = equal_weight_baseline(test_returns)
    mv, mv_w = mean_variance_offline(test_returns)

    print('Equal weight CumRet:', (np.prod(1+ew)-1), 'Sharpe:', sharpe_ratio(ew))
    print('MV CumRet:', (np.prod(1+mv)-1), 'Sharpe:', sharpe_ratio(mv))

    # Plot weights
    plt.figure(figsize=(10,6))
    plt.plot(weights)
    plt.title('Learned portfolio weights over evaluation horizon')
    plt.legend([f'Asset {i}' for i in range(N)])
    plt.show()

    # Save actor
    os.makedirs('models_rl', exist_ok=True)
    torch.save(agent.actor.state_dict(), 'models_rl/ddpg_actor.pth')
    print('Saved actor to models_rl/ddpg_actor.pth')

   
