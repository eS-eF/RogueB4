import sys
import gym
from gym import spaces

import numpy as np
#import MapGenarator
from .MapGenarator import RougeMapGenerator

import random

from gym.envs.registration import register


class RogueEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    N_map = 50
    mapH = 10
    mapW = 20
    N_room = 2
    roomH = 2
    roomW = 2
    MAX_step = 200

    reward_goal = 1.00
    reward_nomove = -1e-9

    map_generator = RougeMapGenerator(
        N_map,
        mapH,
        mapW,
        N_room,
        False,
        roomH,
        roomW,
        False
    )


    def __init__(self):
        super(RogueEnv, self).__init__()

        self.action_space = spaces.Discrete(4)       # エージェントが取りうる行動空間を定義
        self.observation_space = spaces.Box(
            np.zeros((3, self.mapH, self.mapW), dtype=np.float32),
            np.ones((3, self.mapH, self.mapW), dtype=np.float32),
            (3, self.mapH, self.mapW)
        )  # エージェントが受け取りうる観測空間を定義
        ###self.reward_range = ...       # 報酬の範囲[最小値と最大値]を定義

        self.pi = -1
        self.pj = -1 #プレイヤー位置
        self.si = -1
        self.sj = -1 #階段位置
        self.turn = 0
        self.obs = np.zeros(
            (3, self.mapH, self.mapW),
            dtype=np.uint8
        )


        #マップの事前生成を行う
        self.map_generator.Generate()
        

    def reset(self):
        # 環境を初期状態にする関数
        # 初期状態をreturnする
        cur_map, self.si, self.sj, self.pi, self.pj = self.map_generator.GetMap(
            random.randrange(0, self.N_map)
        )
        for i in range(self.mapH):
            for j in range(self.mapW):
                if (i, j) == (self.si, self.sj):
                    self.obs[1][i][j] = 1
                    self.obs[0][i][j] = 1
                elif (i, j) == (self.pi, self.pj):
                    self.obs[2][i][j] = 1
                    self.obs[0][i][j] = 1
                else:
                    self.obs[0][i][j] = 0 if cur_map[i][j] == 0 else 1
                    self.obs[1][i][j] = 0
                    self.obs[2][i][j] = 0

        self.turn = 0

        #print('reset. -------------------------')
        #self.render('human')
        

        return self.obs

    def step(self, action):
        # 行動を受け取り行動後の状態をreturnする

        next_pi = self.pi
        next_pj = self.pj
        reward = 0.00
        done = False
        info = {}

        if action == 0:
            next_pi, next_pj = self.pi, self.pj + 1
        elif action == 1:
            next_pi, next_pj = self.pi + 1, self.pj
        elif action == 2:
            next_pi, next_pj = self.pi, self.pj - 1
        else:
            next_pi, next_pj = self.pi - 1, self.pj
        
        #print('@ (i, j) = ', self.pi, self.pj, ',', 'action = ', action)
        ### ターンの経過
        self.turn += 1

        ### 行く先が階段の場合、クリア
        if self.obs[1][next_pi][next_pj] == 1:
            reward = self.reward_goal
            done = True
            return self.obs, reward, done, info

        ### ターンが上限になったら終了
        if self.turn == self.MAX_step:
            done = True
            reward = 0.00
            return self.obs, reward, done, info
        
        ### マップの外に出る場合、無効
        if not ((0 <= next_pi < self.mapH) and (0 <= next_pj < self.mapW)):
            reward = self.reward_nomove
            return self.obs, reward, done, info
        
        ### 行く先が壁の場合、無効
        if self.obs[0][next_pi][next_pj] == 0:
            reward = self.reward_nomove
            return self.obs, reward, done, info
        


        ### 位置を更新して次へ
        self.obs[2][self.pi][self.pj] = 0
        self.obs[2][next_pi][next_pj] = 1
        self.pi, self.pj = next_pi, next_pj

        reward = 0.00
        done = False

        
        return self.obs, reward, done, info

    def render(self, mode='human'):   
        # modeとしてhuman, rgb_array, ansiが選択可能
        # humanなら描画し, rgb_arrayならそれをreturnし, ansiなら文字列をreturnする
        if mode == 'human':
            for i in range(self.mapH):
                s = ''
                for j in range(self.mapW):
                    if (i, j) == (self.pi, self.pj):
                        s += '@'
                    elif (i, j) == (self.si, self.sj):
                        s += 'G'
                    elif self.obs[0][i][j] == 0:
                        s += '#'
                    else:
                        s += '.'
                print(s)
  
    def close(self):
        pass

    def seed(self, seed=None):
        pass