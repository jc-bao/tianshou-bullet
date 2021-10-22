import gym
from gym import spaces
import numpy as np
import logging
from gym.envs.registration import register
import matplotlib.pyplot as plt
from matplotlib import animation
import time

class NaivePickAndPlace(gym.Env):
    def __init__(self, reward_type='dense_o2g', dim = 2):
        self.dim = dim
        self.reward_type = reward_type
        self._max_episode_steps = 50
        self.space = spaces.Box(low=-np.ones(dim), high=np.ones(dim))
        self.observation_space = spaces.Box(low=-np.ones(dim*2), high=np.ones(dim*2))
        self.action_space = spaces.Box(low=-np.ones(dim+1), high=np.ones(dim+1))
        self.reset()

    def step(self, action):
        self.num_step += 1
        action = np.clip(action, self.space.low, self.space.high)
        self.pos = np.clip(self.pos + action * 0.30, self.space.low, self.space.high)
        obs = np.concatenate((self.pos,self.obj,self.goal)) 
        d = np.linalg.norm(self.obj - self.goal)
        if self.reward_type == 'dense':

        elif self.reward_type == 'dense_o2g':
            reward = -d
        elif self.reward_type == 'sparse':
            reward = (d<0.05).astype(np.float32)
        elif self.reward_type == 'dense_diff_o2g':
            reward = self.d_old - d
            self.d_old = d
        else:
            raise NotImplementedError
        info = {
            'is_success': (d < 0.05),
        }
        done = (self.num_step >= self._max_episode_steps) or (d < 0.05)
        return obs, reward, done, info

    def reset(self):
        self.num_step = 0
        self.goal = self.space.sample()
        self.obj = self.space.sample()
        self.pos = self.space.sample()
        self.d_old = np.linalg.norm(self.pos - self.goal)
        return self.pos

    def render(self):
        if self.num_step == 1:
            # self.fig = plt.figure()
            # self.ax = self.fig.add_subplot()
            self.x = [self.pos[0]]
            self.y = [self.pos[1]]
        self.x.append(self.pos[0])
        self.y.append(self.pos[1])
        if self.num_step == self._max_episode_steps:
            for i in range(len(self.x)):
                plt.plot(self.x[i], self.y[i], 'o', color = [0,0,1,i/50])
            plt.plot(self.goal[0], self.goal[1], 'rx')
            plt.show()
            '''
            fig = plt.figure()
            ax = fig.add_subplot()
            point, = ax.plot([self.x[0]], [self.y[0]], 'o')
            def update_point(n, x, y, point):
                point.set_data(np.array([self.x[n], self.y[n]]))
                return point
            ani=animation.FuncAnimation(fig, func = update_point, frames = 49, interval = 1/30, fargs=(self.x, self.y, point))
            '''

register(
    id='NaivePickAndPlace-v0',
    entry_point='test_env.naive_pac:NaivePickAndPlace',
)