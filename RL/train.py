"""
main training script

features:
mixed precision amp on MPS
4 sgd updates per env step
n step / lambda return
curriculum
deterministic evaluation
sime checkpoint
"""

import os
import time
import json
import numpy as np
import torch
from tqdm import tqdm

from env import CryptoTradingEnvLongShort
from agent import DuelingDQNAgent
from utils import AdvancedLogger

def compute_n_step_target(rews: torch.Tensor, next_q: torch.Tensor,
                          done: torch.Tensor, gamma=0.99, n=1, lam=0.8):
    if n == 1:
        return rews[:, 0] + gamma * next_q.max(1)[0] * (1.0 - done)
    
    boot = torch.sum(gamma**torch.arange(n).to(rews.device)*rews, dim=1)
    
    mix = lam*next_q.max(1)[0] * (gamma**n) * (1.0 - done)
    return boot + mix

def train(
    norm_path: str = "norm.npy",
    raw_path: str = "raw.npy",
    episodes: int = 1000,
    device: str = "mps",
    batch_size: int = 256,
    eval_every: int = 50,
    warmup_episodes: int = 100
):
    norm = np.load(norm_path)
    raw = np.load(raw_path)
    
    env = CryptoTradingEnvLongShort(norm,raw,init_balance=10_000,fee_pct=0.001,episode_len=500,random_start=True,lookback=10)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DuelingDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden=128,
        lr=5e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        per_alpha=0.6,
        per_capacity=200_000
    )
    
    logger = AdvancedLogger()
    returns = []
    best_mean_50 = -1e9
    breakthrough_episode = None
    
    for epi in tqdm(range(episodes), desc="Training"):
        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        ep_reward = 0.0
        
        while not done:
            mask = env._valid_mask()
            
            action = agent.act(obs, mask)
            
            next_obs, reward, done, info = env.step(action)
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=agent._device).unsqueeze(0)
            
            agent.remember(obs.cpu().numpy()[0], action, float(reward), next_obs.cpu().numpy()[0], float(done))
            
            obs = next_obs
            ep_reward += reward
            
            if epi >= warmup_episodes and len(agent.memory) >= batch_size:
                for _ in range(4):
                    agent.replay(batch_size)
                        
        returns.append(ep_reward/env.init_balance)
        
        if epi >= warmup_episodes:
            agent.epsilon_decay()
        
        if (epi + 1) % agent.update_target_every == 0:
            agent.update_target()
            
        if (epi + 1) % eval_every == 0:
            eval_metrics = evaluate(env, agent, n_episodes=10, device=device)
            logger.log_evaluation(epi + 1, eval_metrics)
            
            logger.logger.info("\n" + "=" * 60)
            logger.logger.info(f"EVAL @ episode {epi+1}")
            logger.logger.info(f" Mean Return: {eval_metrics['mean_return']:.2%}")
            logger.logger.info(f" Mean Sharpe: {eval_metrics['mean_sharpe']:.3%}")
            logger.logger.info(f" Mean WinRate: {eval_metrics['mean_win_rate']:.2%}")
            logger.logger.info("=" * 60 + "\n")
            
        if (epi + 1) % 10 == 0:
            avg_50 = np.mean(returns[-50:] if len(returns) >= 50 else np.mean(returns))
            if avg_50 > best_mean_50 * 1.15 and breakthrough_episode is None:
                best_mean_50 = avg_50
                breakthrough_episode = epi + 1
                logger.logger.info("\n" + "=" * 60)
                logger.logger.info(f"Breakthrough Episode @ {breakthrough_episode} - 50 episode average = {avg_50:.2%}")
                
        ep_info = env.get_episode_data()
        ep_info["final_return"] = returns[-1]
        logger.log_episode_details(epi+1, ep_info, threshold=0.08)
                
    final_path = os.path.join(logger.run_dir, f"model_final_{int(time.time())}.pth")
    agent.save(final_path)
    logger.logger.info(f"Final Model saved to {final_path}")
    
    logger.plot_results(returns, logger.eval_results)
    
    summary = {
        "total_episodes": episodes,
        "avg_returns_all": float(np.mean(returns)),
        "avg_returns_last_50": float(np.mean(returns[-50:])),
        "best_return": float(np.max(returns)),
        "worst_return": float(np.min(returns)),
        "win_rate_all": float(np.mean([r > 0 for r in returns])),
        "breakthrough_episode": breakthrough_episode,
        "final_beta": float(agent.beta),
        "model_path": final_path,
        "run_dir": logger.run_dir,
        "evaluation_results": logger.eval_results
    }
    
    with open(os.path.join(logger.run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    logger.logger.info("\n"*60)
    logger.logger.info("TRAINING COMPLETE")
    
    for k,v in summary.items():
        logger.logger.info(f" {k}: {v}")
    logger.logger.info("\n"*60)
    
    return returns, logger
        
def evaluate(env: CryptoTradingEnvLongShort, agent: DuelingDQNAgent, n_episodes: int = 10, device: str = "msp"):
    eps_old = agent.epsilon if hasattr(agent, "epsilon") else None
    if eps_old is not None:
        agent.epsilon = 0.0
        
    rets, sharpe, win = [],[],[]
    for _ in range(n_episodes):
        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        
        while not done:
            mask = env._valid_mask()
            q = agent.policy(obs).cpu().numpy()[0]
            q[mask == 0] = -np.inf
            action = int(np.argmax(q))
            nxt, reward, done, info = env.setp(action)
            obs = torch.tensor(nxt, dtype=torch.float32, device=device).unsqueeze(0)
            
        rets.append(info["portfolio_return"])
        sharpe.append(info("sharpe_ratio"))
        win.append(info["positive_trades"] / max(info["total_trades"], 1))
        
    if eps_old is not None:
        agent.epsilon = eps_old
        
    return {
        "mean_return": float(np.mean(rets)),
        "std_return": float(np.std(rets)),
        "mean_sharpe": float(np.mean(sharpe)),
        "mean_win_rate": float(np.mean(win)),
        "best_return": float(np.max(rets)),
        "worst_return": float(np.min(rets))
    }
    
if __name__ == "__main__":
    train()
    