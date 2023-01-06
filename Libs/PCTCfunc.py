import logging
import numpy as np
import gym
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


### minihack を one-hot にする
class MH_Obs_OneHot_Wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.K = 4
        lo = np.array([[[0 for i in range(79)] for j in range(21)] for k in range(self.K)])
        hi = np.array([[[1 for i in range(79)] for j in range(21)] for k in range(self.K)])
        self.observation_space = gym.spaces.Box(lo, hi, (self.K, 21, 79), np.int16)
    def observation(self, observation):
        obs = observation['glyphs']
        new_obs = np.zeros((self.K, 21, 79))
        for i in range(len(obs)):
            for j in range(len(obs[i])):
                ch = chr(observation['chars'][i][j])
                if ch == '.' or ch == '<' or ch == '+':
                    new_obs[1][i][j] = 1
                elif ch == '-' and observation['colors'][i][j] == 3:
                    new_obs[1][i][j] = 1
                elif ch == '@':
                    new_obs[2][i][j] = 1
                elif ch == '>':
                    new_obs[3][i][j] = 1
                else:
                    new_obs[0][i][j] = 1
        return new_obs


### PCTCラッパー
class PCTC_CuriosityWrapper(VecEnvWrapper):
    """
    Test : Random Network Distillation + TC
    """

    def __init__(self, 
    venv: VecEnv,
    network: str = 'mlp',
    intrinsic_reward_weight: float = 1.00,
    intrinsic_reward_weight_2: float = 0.10,
    buffer_size: int = 65535,
    train_freq: int = 1024,
    gradient_steps: int = 4,
    batch_size: int = 512,
    learning_starts: int = 100,
    filter_end_of_episode: bool = True,
    filter_reward: bool = False,
    norm_obs: bool = True,
    norm_ext_reward: bool = True,
    gamma: float = 0.99,
    learning_rate: float = 0.0001,
    training: bool = True
    ):
        # initialize parameters
        super().__init__(venv)
        self.network_type = network
        self.buffer = ReplayBuffer(buffer_size, self.observation_space, self.action_space, handle_timeout_termination=False)
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.intrinsic_reward_weight_2 = intrinsic_reward_weight_2
        self.filter_end_of_episode = filter_end_of_episode
        self.filter_extrinsic_reward = filter_reward
        self.clip_obs = 5
        self.norm_obs = norm_obs
        self.norm_ext_reward = norm_ext_reward
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.training = training

        self.epsilon = 1e-8
        self.int_rwd_rms = RunningMeanStd(shape=(), epsilon=self.epsilon)
        self.ext_rwd_rms = RunningMeanStd(shape=(), epsilon=self.epsilon)
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.int_ret = (np.zeros(self.num_envs)) # discounted return for intrinsic reward
        self.ext_ret = (np.zeros(self.num_envs)) # discounted return for extrinsic reward

        self.int_rwd_rms_2 = RunningMeanStd(shape=(), epsilon=self.epsilon)
        self.int_ret_2 = np.zeros(self.num_envs)

        self.updates = 0
        self.steps = 0
        self.last_action = None
        self.last_obs = None
        self.last_update = 0


        self.input_dim = 1
        for i in range(len(self.observation_space.shape)):
            self.input_dim *= self.observation_space.shape[i]
        

        self.predictor_network = None
        self.target_network = None
        self.params = None
        self.int_reward = None
        self.aux_loss = None
        self.optimimzer = None
        self.training_op = None

        self.loss_fn = None

        self.feature_network = None
        self.buflen = 128
        self.PCTC_buffer = deque(maxlen=self.buflen)
        self.gamma_2 = 0.99
        

        self.setup_model()
    
    def setup_model(self):
        # initialize model for RND
        self.target_network = nn.Sequential(
            nn.Conv2d(3, 16, 3), # 19 * 77 * 16
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), #10 * 39 * 16
            nn.Flatten(),
            nn.Linear(576, 512)
        )
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.predictor_network = nn.Sequential(
            nn.Conv2d(3, 16, 3), # 19 * 77 * 16
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2), #10 * 39 * 16
            nn.Flatten(),
            nn.Linear(576, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.optimimzer = optim.Adam(self.predictor_network.parameters(), lr = self.learning_rate)

        self.loss_fn = F.mse_loss
        self.loss_fn_l = F.mse_loss

        self.feature_network = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(576, 512)
        )
        for param in self.feature_network.parameters():
            param.requires_grad = False


    
    def reset(self):
        obs = self.venv.reset()
        self.last_obs = obs
        return obs
    
    def step_async(self, actions: np.ndarray):
        super().step_async(actions)
        self.last_action = actions
        self.steps += self.num_envs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()

        #print(type(self.last_obs))
        #print(type(self.last_action))
        #print(type(rews))
        #print(type(obs))
        #print(type(dones))
        #print(self.last_obs.shape, self.last_action.shape, rews.shape, obs.shape, dones.shape)
        
        self.buffer.extend(self.last_obs, obs, self.last_action, rews, dones, infos)
        

        if self.filter_extrinsic_reward:
            rews = np.zeros(rews.shape)
        if self.filter_end_of_episode:
            dones = np.zeros(dones.shape)
        
        if self.training:
            self.obs_rms.update(obs)
        
        obs_n = torch.tensor(self.normalize_obs(obs), dtype=torch.float32)
        #print(obs_n, type(obs_n))
        y_targ = self.target_network(obs_n).detach()
        y_pred = self.predictor_network(obs_n)
        f_vec  = self.feature_network(obs_n).detach()

        loss = self.loss_fn(y_targ, y_pred)

        loss_np = loss.detach().cpu().numpy().copy() #update for GPU


        ## 提案 報酬の計算
        self.PCTC_buffer.append(f_vec)

        
        int_rew = 0
        if len(self.PCTC_buffer) == self.buflen:
            for i in range(self.buflen - 1):
                int_rew = int_rew * self.gamma_2 + np.linalg.norm(self.PCTC_buffer[-i] - self.PCTC_buffer[-i-1])


        if self.training:
            self._update_ext_reward_rms(rews)
            self._update_int_reward_rms(loss_np)
            self._update_int_reward_rms_2(int_rew)


        
        intirinsic_reward = np.array(loss_np) / np.sqrt(self.int_rwd_rms.var + self.epsilon)
        intirinsic_reward_2 = np.array(int_rew) / np.sqrt(self.int_rwd_rms_2.var + self.epsilon)

        if self.norm_ext_reward:
            extrinsic_reward = np.array(rews) / np.sqrt(self.ext_rwd_rms.var + self.epsilon)
        else:
            extrinsic_reward = rews

        reward = np.array(extrinsic_reward + self.intrinsic_reward_weight * intirinsic_reward + self.intrinsic_reward_weight_2 * intirinsic_reward_2)
        

        if self.training and self.steps > self.learning_starts and self.steps - self.last_update > self.train_freq:
            self.updates += 1
            self.last_update = self.steps
            self.learn()
            print('EX : ', extrinsic_reward, 'INT : ',self.intrinsic_reward_weight*intirinsic_reward , 'INT2 : ', self.intrinsic_reward_weight_2*intirinsic_reward_2)
        
        
        #print('reward : ', reward)
        
        return obs, reward, dones, infos
            
    def _update_int_reward_rms(self, reward: np.ndarray) -> None:
        """Update reward normalization statistics."""
        self.int_ret = self.gamma * self.int_ret + reward
        self.int_rwd_rms.update(self.int_ret)

    def _update_ext_reward_rms(self, reward: np.ndarray) -> None:
        """Update reward normalization statistics."""
        self.ext_ret = self.gamma * self.ext_ret + reward
        self.ext_rwd_rms.update(self.ext_ret)
    
    def _update_int_reward_rms_2(self, reward: np.ndarray) -> None:
        self.int_ret_2 = self.gamma * self.int_ret_2 + reward
        self.int_rwd_rms_2.update(self.int_ret_2)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observations using observations statistics.
        Calling this method does not update statistics.
        """
        if self.norm_obs:
            obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                          -self.clip_obs,
                          self.clip_obs)
        return obs
    
    def close(self):
        VecEnvWrapper.close(self)
    def learn(self):
        total_loss = 0
        for _ in range(self.gradient_steps):
            obs_batch, act_batch, rews_batch, next_obs_batch, done_mask = self.buffer.sample(self.batch_size)
            obs_batch = torch.tensor(self.normalize_obs(obs_batch), dtype=torch.float32, requires_grad=False)
            y_targ = self.target_network(obs_batch).detach()
            y_pred = self.predictor_network(obs_batch)

            self.optimimzer.zero_grad()
            loss = self.loss_fn_l(y_targ, y_pred)
            self.optimimzer.step()
            total_loss += loss
        print("Trained predictor. Avg loss: {}".format(total_loss / self.gradient_steps))
