import argparse
from environment import *
from stable_baselines3 import PPO
from PPO_agent import NavigatorAC

if __name__ == "__main__":
    # Environment
    env = Environment("./environment.xml")
    
    # Agent
    model = PPO(NavigatorAC, env, verbose=1, batch_size=1024, gamma=0.99, device="cuda", seed=42)
    agent = model.load("./dynamic_sb_ckpt/navigator_3150000_steps.zip")
    
    # Play
    env.render(agent)
