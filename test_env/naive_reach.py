import gym
from gym import spaces
import numpy as np
import logging
from gym.envs.registration import register
import matplotlib.pyplot as plt
from matplotlib import animation
import time

from numpy.core import einsumfunc

class NaiveReach(gym.Env):
    def __init__(self, reward_type='dense_diff', dim = 2):
        self.dim = dim
        self.reward_type = reward_type
        self._max_episode_steps = 50
        self.space = spaces.Box(low=-np.ones(dim), high=np.ones(dim))
        self.observation_space = spaces.Box(low=-np.ones(dim*2), high=np.ones(dim*2))
        self.action_space = spaces.Box(low=-np.ones(dim), high=np.ones(dim))
        self.reset()

    def step(self, action):
        self.num_step += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.pos = np.clip(self.pos + action * 0.20, self.space.low, self.space.high)
        obs = np.concatenate((self.pos, self.goal)) 
        d = np.linalg.norm(self.pos - self.goal)
        if self.reward_type == 'dense':
            reward = -d
        elif self.reward_type == 'sparse':
            reward = (d<0.05).astype(np.float32)
        elif self.reward_type == 'dense_diff':
            reward = self.d_old - d
            self.d_old = d
        info = {
            'is_success': (d < 0.05),
        }
        done = (self.num_step >= self._max_episode_steps) or (d < 0.05)
        return obs, reward, done, info

    def reset(self):
        self.num_step = 0
        self.goal = self.space.sample()
        self.pos = self.space.sample()
        self.d_old = np.linalg.norm(self.pos - self.goal)
        return np.concatenate((self.pos, self.goal)) 

    def render(self):
        if self.num_step == 1:
            self.data = [self.pos]
        self.data.append(self.pos)
        if self.num_step == self._max_episode_steps or (np.linalg.norm(self.pos - self.goal))<0.05:
            if self.dim == 2:
                for i,d in enumerate(self.data):
                    plt.plot(d[0], d[1], 'o', color = [0,0,1,i/50])
                plt.plot(self.goal[0], self.goal[1], 'rx')
                plt.show()
            elif self.dim == 3:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                for i,d in enumerate(self.data):
                    ax.scatter(d[0], d[1], d[2], 'o', color = [0,0,1,i/50])
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

    def ezpolicy(self, obs):
        return obs[self.dim:] - obs[:self.dim]

register(
    id='NaiveReach-v0',
    entry_point='test_env.naive_reach:NaiveReach',
)

if __name__ == '__main__':
    env = NaiveReach()
    obs = env.reset()
    for i in range(50):
        act = env.ezpolicy(obs)
        obs, reward, done, info = env.step(act)
        env.render()
        print('[obs, reward, done]', obs, reward, done)
