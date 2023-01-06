import Envs
import gym
import random
import numpy as np
import sys

from Libs import RNDfunc, PCTCfunc, EVALfunc

import logging
import os
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


def Exp_Run(
    envname = 'myrogue-v0',
    learn_timestep = 5000000,
    exp_code = 0
    ):

    print('START')
    ### 使用する環境の設定
    env = DummyVecEnv([lambda : gym.make(envname)])
    
    log_dir = '../dummy/logs/' + str(exp_code)
    os.makedirs(log_dir, exist_ok=True)
    ### Monitor の設定
    env = VecMonitor(env, log_dir)

    ### 好奇心ラッパーの適用
    env = RNDfunc.RND_CuriosityWrapper(env, 
    intrinsic_reward_weight=0.005, 
    norm_ext_reward=True, 
    filter_end_of_episode=False)

    ### モデルの用意、学習
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=learn_timestep)


    model_dir = '../dummy/models/' + str(exp_code)
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)

    print('END')


### コード実行部分

def main():
    args = sys.argv
    num = 0
    if len(args) >= 2:
        num = int(args[1])
    Exp_Run(exp_code=num)

if __name__ == '__main__':
    main()


'''
Todo: 
    モデル、モニターの記録先の設定をより簡単に
    ハイパーパラメータを可能な限り run.py で渡すようにする
'''