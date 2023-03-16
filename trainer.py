from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gym import spaces
import torch as th
from torch import nn

from torch.utils.tensorboard import SummaryWriter
from environment import Environment
from PPO_agent import NavigatorAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback



if __name__ == "__main__":

    env = Environment("./environment.xml")
    check_env(env)
    checkpoint_callback = CheckpointCallback(
            save_freq = 50000,
            save_path = "./dynamic_sb_ckpt/",
            name_prefix = "navigator",
        )
    
    
    model = PPO(NavigatorAC, env, verbose=1, batch_size=1024, gamma=0.99, device="cuda", seed=42, tensorboard_log="sb_dynamic_ppo")
    model.learn(10000000, callback=checkpoint_callback)
