import logging
import numpy as np
import gym
import os
import minihack
from nle import nethack
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stable_baselines3.common import torch_layers, utils
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper, VecEnv, VecEnvStepReturn
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor

from stable_baselines3 import PPO

import pandas as pd
import matplotlib.pyplot as plt

def evaluate(model, env, num_episodes=100):
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    return mean_episode_reward