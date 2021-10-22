import gym
from gym import spaces
import numpy as np
import logging
from gym.envs.registration import register
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import os

class NaivePickAndPlace(gym.Env):
    def __init__(self, reward_type='dense', dim = 2, error = 0.05, mode = 'static'):
        self.dim = dim
        self.reward_type = reward_type
        self._max_episode_steps = 50
        self.error = error
        self.save_num = 0
        '''
        mode:
        'static': the object is still and not movable
        'dynamic': the object can be moved with input `action`
        '''
        self.mode = mode
        self.space = spaces.Box(low=-np.ones(dim), high=np.ones(dim))
        self.observation_space = spaces.Box(low=-np.ones(dim*3), high=np.ones(dim*3))
        if self.mode == 'static':
            '''
            Action Space
            [0:dim] agent1 position
            [dim] agent1 gripper
            '''
            self.action_space = spaces.Box(low=-np.ones(dim+1), high=np.ones(dim+1))
        elif self.mode == 'dynamic':
            '''
            Action Space
            [0:dim] agent1 position
            [dim] agent1 gripper
            [dim+1:dim*2+1] object position
            '''
            self.action_space = spaces.Box(low=-np.ones(dim*2+1), high=np.ones(dim*2+1))
        self.reset()

    def step(self, action):
        # environment step
        self.num_step += 1
        self.if_grasp = (action[self.dim] < 0) and self.d_a2o < self.error
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.pos = np.clip(self.pos + action[:self.dim] * 0.10, self.space.low, self.space.high)
        if self.if_grasp:
            self.obj = self.pos
        elif self.mode == 'dynamic':
            # in dynamic mode, object can move with action
            self.obj = np.clip(self.obj + action[self.dim+1:] * 0.10, self.space.low, self.space.high)
        self.d_o2g = np.linalg.norm(self.obj - self.goal)
        self.d_a2o = np.linalg.norm(self.pos - self.obj)
        # get obs
        obs = np.concatenate((self.pos,self.obj,self.goal)) 
        # get reward
        if self.reward_type == 'dense':
            if not self.if_grasp:
                reward = 0.5 * (1 - np.tanh(2.0 * self.d_a2o))
            else:
                reward = (0.5 + 0.5*(1 - np.tanh(1.0 * self.d_o2g)))
        elif self.reward_type == 'dense_o2g':
            reward = -self.d_o2g
        elif self.reward_type == 'sparse':
            reward = (self.d_o2g<self.error).astype(np.float32)
        elif self.reward_type == 'dense_diff_o2g':
            reward = self.d_old - self.d_o2g
            self.d_old = self.d_o2g
        else:
            raise NotImplementedError
        info = {
            'is_success': (self.d_o2g < self.error),
        }
        done = (self.num_step >= self._max_episode_steps) or (self.d_o2g < self.error)
        return obs, reward, done, info

    def reset(self):
        self.num_step = 0
        self.if_grasp = False
        self.goal = self.space.sample()
        self.obj = self.space.sample()
        self.pos = self.space.sample()
        self.d_old = np.linalg.norm(self.pos - self.goal)
        self.d_a2o = np.linalg.norm(self.pos - self.obj)
        self.d_o2g = np.linalg.norm(self.obj - self.goal)
        return np.concatenate((self.pos,self.obj,self.goal)) 

    def render(self, mode = 'human', save_path = None):
        if self.num_step == 1:
            self.pos_data = [self.pos]
            self.obj_data = [self.obj]
        self.pos_data.append(self.pos)
        self.obj_data.append(self.obj)
        if self.num_step == self._max_episode_steps or (np.linalg.norm(self.pos - self.goal))<self.error:
            if self.dim == 2:
                for i,d in enumerate(self.pos_data):
                    plt.plot(d[0], d[1], 'o', color = [0,0,1,i/50])
                for i,d in enumerate(self.obj_data):
                    plt.plot(d[0], d[1], 'o', markersize = 2.5, color = [0,1,0,i/50])
                plt.plot(self.goal[0], self.goal[1], 'rx')
                if mode == 'human':
                    plt.show()
                    plt.clf()
                elif mode == 'save':
                    fullpath = os.path.join(save_path, 'pic')
                    if not os.path.exists(fullpath):
                        os.makedirs(fullpath)
                    fullpath = os.path.join(save_path, 'pic/pic'+str(self.save_num)+'.png')
                    plt.savefig(fullpath)
                    self.save_num += 1
                    plt.clf()
            elif self.dim == 3:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                for i,d in enumerate(self.pos_data):
                    ax.scatter(d[0], d[1], d[2], 'o', color = [0,0,1,i/50])
                for i,d in enumerate(self.obj_data):
                    ax.scatter(d[0], d[1], d[2], 'o', s = 6, color = [0,1,0,i/50])
                if mode == 'human':
                    plt.show()
                    plt.clf()
                elif mode == 'save':
                    fullpath = os.path.join(save_path, 'pic')
                    if not os.path.exists(fullpath):
                        os.makedirs(fullpath)
                    fullpath = os.path.join(save_path, 'pic/pic'+str(self.save_num)+'.png')
                    plt.savefig(fullpath)
                    self.save_num += 1
                    plt.clf()

    def ezpolicy(self, obs):
        pos = obs[:self.dim]
        obj = obs[self.dim:self.dim*2]
        goal = obs[self.dim*2:]
        if_reach = np.linalg.norm(pos - obj)< self.error
        if not if_reach:
            act =  np.append((obj - pos),1)
        else:
            act = np.append((goal - pos),-1)
        if self.mode == 'dynamic':
            act = np.append(act, pos - obj)
        return act

register(
    id='NaivePickAndPlace-v0',
    entry_point='test_env.naive_pac:NaivePickAndPlace',
)

if __name__ == '__main__':
    env = NaivePickAndPlace()
    obs = env.reset()
    rew_data = []
    for i in range(50):
        act = env.ezpolicy(obs)
        obs, reward, done, info = env.step(act)
        env.render()
        rew_data.append(reward)
        print('[obs, reward, done]', obs, reward, done)
    plt.plot(np.arange(50), rew_data)
    plt.show()