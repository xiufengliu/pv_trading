"""
Reinforcement Learning Agent for PV Intraday Trading

This module implements a PPO (Proximal Policy Optimization) agent for learning
optimal trading policies in the intraday market environment.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from market_environment import IntradayMarketEnvironment
import warnings
warnings.filterwarnings('ignore')


class TradingCallback(BaseCallback):
    """Custom callback for monitoring training progress"""
    
    def __init__(self, eval_env, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_rewards = []
        self.eval_profits = []
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current policy
            obs = self.eval_env.reset()
            total_reward = 0
            total_profit = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                total_reward += reward
                if 'profit' in info:
                    total_profit += info['profit']
            
            self.eval_rewards.append(total_reward)
            self.eval_profits.append(total_profit)
            
            if self.verbose > 0:
                print(f"Eval at step {self.n_calls}: Reward={total_reward:.4f}, Profit={total_profit:.4f}")
        
        return True


class PPOTradingAgent:
    """PPO-based trading agent for PV intraday trading"""
    
    def __init__(self, 
                 env: IntradayMarketEnvironment,
                 policy: str = 'MlpPolicy',
                 learning_rate: float = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 ent_coef: float = 0.01,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 device: str = 'auto'):
        
        self.env = env
        self.policy = policy
        
        # PPO hyperparameters
        self.hyperparams = {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_range': clip_range,
            'ent_coef': ent_coef,
            'vf_coef': vf_coef,
            'max_grad_norm': max_grad_norm,
            'device': device
        }
        
        self.model = None
        self.training_history = []
    
    def train(self, 
              total_timesteps: int = 100000,
              eval_env: Optional[IntradayMarketEnvironment] = None,
              eval_freq: int = 5000,
              save_path: Optional[str] = None) -> 'PPOTradingAgent':
        """Train the PPO agent"""
        
        # Create vectorized environment
        vec_env = DummyVecEnv([lambda: self.env])
        
        # Initialize PPO model
        self.model = PPO(
            self.policy,
            vec_env,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            **self.hyperparams
        )
        
        # Setup evaluation callback
        callback = None
        if eval_env is not None:
            callback = TradingCallback(eval_env, eval_freq=eval_freq)
        
        # Train the model
        print(f"Starting PPO training for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        # Save training history
        if callback is not None:
            self.training_history = {
                'eval_rewards': callback.eval_rewards,
                'eval_profits': callback.eval_profits
            }
        
        # Save model
        if save_path is not None:
            self.model.save(save_path)
            print(f"Model saved to {save_path}")
        
        return self
    
    def predict(self, 
                observation: np.ndarray, 
                deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make prediction using trained model"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        return self.model.predict(observation, deterministic=deterministic)
    
    def evaluate(self, 
                 eval_env: IntradayMarketEnvironment,
                 n_episodes: int = 10,
                 deterministic: bool = True) -> Dict[str, Any]:
        """Evaluate the trained agent"""
        
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        episode_rewards = []
        episode_profits = []
        episode_summaries = []
        
        for episode in range(n_episodes):
            obs = eval_env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = eval_env.step(action)
                total_reward += reward
            
            episode_rewards.append(total_reward)
            summary = eval_env.get_episode_summary()
            episode_summaries.append(summary)
            
            if 'total_profit' in summary:
                episode_profits.append(summary['total_profit'])
        
        # Aggregate results
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_profit': np.mean(episode_profits) if episode_profits else 0,
            'std_profit': np.std(episode_profits) if episode_profits else 0,
            'episode_rewards': episode_rewards,
            'episode_profits': episode_profits,
            'episode_summaries': episode_summaries
        }
        
        return results
    
    def get_trading_decisions(self, 
                             features: pd.DataFrame,
                             deterministic: bool = True) -> pd.DataFrame:
        """Get trading decisions for a sequence of features"""
        
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        decisions = []
        
        for i in range(len(features)):
            obs = features.iloc[i].values.astype(np.float32)
            action, _ = self.model.predict(obs, deterministic=deterministic)
            
            # Parse action
            action_type = int(np.round(action[0]))
            volume_fraction = np.clip(action[1], 0, 1)
            
            action_names = {0: 'hold', 1: 'buy', 2: 'sell'}
            
            decisions.append({
                'datetime': features.index[i],
                'action_type': action_names.get(action_type, 'hold'),
                'action_type_num': action_type,
                'volume_fraction': volume_fraction,
                'raw_action': action
            })
        
        return pd.DataFrame(decisions).set_index('datetime')
    
    def load_model(self, path: str) -> 'PPOTradingAgent':
        """Load a pre-trained model"""
        self.model = PPO.load(path)
        return self
    
    def get_training_history(self) -> Dict[str, List]:
        """Get training history"""
        return self.training_history


class CustomPPOPolicy(nn.Module):
    """Custom neural network policy for PPO"""
    
    def __init__(self, 
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 hidden_dims: List[int] = [256, 256],
                 activation: nn.Module = nn.ReLU):
        
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Feature extractor
        input_dim = observation_space.shape[0]
        layers = []
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Policy head (actor)
        self.policy_head = nn.Linear(prev_dim, action_space.shape[0])
        
        # Value head (critic)
        self.value_head = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)
    
    def forward(self, observations):
        features = self.feature_extractor(observations)
        
        # Policy output (mean of action distribution)
        policy_output = self.policy_head(features)
        
        # Value output
        value_output = self.value_head(features)
        
        return policy_output, value_output


def create_training_environment(processor, 
                               train_data: pd.DataFrame,
                               episode_length: int = 168) -> IntradayMarketEnvironment:
    """Create training environment from processed data"""
    
    train_market = processor.market_data.loc[train_data.index]
    train_pv = processor.pv_data.loc[train_data.index]
    
    env = IntradayMarketEnvironment(
        market_data=train_market,
        pv_data=train_pv,
        features=train_data,
        episode_length=episode_length
    )
    
    return env


def hyperparameter_tuning(env: IntradayMarketEnvironment,
                         n_trials: int = 20,
                         n_timesteps: int = 50000) -> Dict[str, Any]:
    """Perform hyperparameter tuning using Optuna"""
    
    try:
        import optuna
    except ImportError:
        print("Optuna not installed. Skipping hyperparameter tuning.")
        return {}
    
    def objective(trial):
        # Suggest hyperparameters
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        n_epochs = trial.suggest_int('n_epochs', 5, 20)
        gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
        gae_lambda = trial.suggest_uniform('gae_lambda', 0.8, 0.99)
        clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)
        ent_coef = trial.suggest_loguniform('ent_coef', 1e-4, 1e-1)
        
        # Create agent with suggested hyperparameters
        agent = PPOTradingAgent(
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef
        )
        
        # Train agent
        agent.train(total_timesteps=n_timesteps)
        
        # Evaluate agent
        results = agent.evaluate(env, n_episodes=5)
        
        return results['mean_profit']
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'study': study
    }


if __name__ == "__main__":
    # Example usage
    from data_processing import load_danish_data
    
    # Load data
    processor = load_danish_data()
    train_data, test_data = processor.get_training_data()
    
    # Create environments
    train_env = create_training_environment(processor, train_data)
    test_env = create_training_environment(processor, test_data)
    
    # Create and train agent
    agent = PPOTradingAgent(train_env)
    agent.train(
        total_timesteps=50000,
        eval_env=test_env,
        eval_freq=5000,
        save_path="models/ppo_trading_agent"
    )
    
    # Evaluate agent
    results = agent.evaluate(test_env, n_episodes=10)
    
    print("Evaluation Results:")
    for key, value in results.items():
        if key not in ['episode_rewards', 'episode_profits', 'episode_summaries']:
            print(f"  {key}: {value:.4f}")
    
    # Get trading decisions
    decisions = agent.get_trading_decisions(test_data)
    print(f"\nTrading decisions shape: {decisions.shape}")
    print(decisions.head())
