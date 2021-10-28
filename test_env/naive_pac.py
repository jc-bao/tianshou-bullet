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
    def __init__(self, config = None): 
        '''
        param
        'use_grasp': if enable grasp action
        '''
        self.dim = config['dim']
        self.reward_type = config['reward_type']
        self._max_episode_steps = 50
        self.error = config['error']
        self.use_grasp = config['use_grasp']
        self.vel = config['vel']
        self.init_grasp_rate = config['init_grasp_rate']
        self.save_num = 0
        self.use_her = config['use_her']
        '''
        mode:
        'static': the object is still and not movable
        'dynamic': the object can be moved with input `action`
        '''
        self.mode = config['mode']
        self.space = spaces.Box(low=-np.ones(self.dim), high=np.ones(self.dim))
        self.observation_space = spaces.Box(low=-np.ones(self.dim*3), high=np.ones(self.dim*3))
        if self.mode == 'static':
            '''
            Action Space
            [0:dim] agent1 position
            [dim] agent1 gripper
            '''
            self.action_space = spaces.Box(low=-np.ones(self.dim+1), high=np.ones(self.dim+1))
        elif self.mode == 'dynamic':
            '''
            Action Space
            [0:dim] agent1 position
            [dim] agent1 gripper
            [dim+1:dim*2+1] object position
            '''
            self.action_space = spaces.Box(low=-np.ones(self.dim*2+1), high=np.ones(self.dim*2+1))
        self.reset()

    def step(self, action):
        # environment step
        self.num_step += 1
        self.if_grasp = ((action[self.dim] < 0) and self.d_a2o < self.error) if self.use_grasp else self.d_a2o < self.error
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.pos = np.clip(self.pos + action[:self.dim] * self.vel, self.space.low, self.space.high)
        if self.if_grasp:
            self.obj = self.pos
        elif self.mode == 'dynamic':
            # in dynamic mode, object can move with action
            self.obj = np.clip(self.obj + action[self.dim+1:] * self.vel, self.space.low, self.space.high)
        self.d_o2g = np.linalg.norm(self.obj - self.goal)
        self.d_a2o = np.linalg.norm(self.pos - self.obj)
        # get obs
        obs = self._get_obs()
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
            'future_length': self._max_episode_steps - self.num_step,
        }
        done = self._get_done()
        return obs, reward, done, info

    def reset(self):
        self.num_step = 0
        self.if_grasp = False
        self.goal = self.space.sample()
        self.pos = self.space.sample()
        if np.random.random() < self.init_grasp_rate:
            self.obj = self.pos
        else: 
            self.obj = self.space.sample()
        self.d_old = np.linalg.norm(self.pos - self.goal)
        self.d_a2o = np.linalg.norm(self.pos - self.obj)
        self.d_o2g = np.linalg.norm(self.obj - self.goal)
        return self._get_obs()

    def render(self, mode = 'image', save_path = None, writer = None):
        print('rendering ...')
        if mode == 'tensorboard':
            for i in range(self.dim):
                writer.add_scalar('render'+str(self.save_num)+'/pos-goal'+str(i), self.pos[i]-self.goal[i], self.num_step)
                writer.add_scalar('render'+str(self.save_num)+'/pos-obj'+str(i), self.pos[i]-self.obj[i], self.num_step)
                writer.add_scalar('render'+str(self.save_num)+'/obj-goal'+str(i), self.obj[i]-self.goal[i], self.num_step)
            if self.done:
                self.save_num += 1
        else:
            if self.num_step == 1:
                self.pos_data = [self.pos]
                self.obj_data = [self.obj]
            self.pos_data.append(self.pos)
            self.obj_data.append(self.obj)
            if self._get_done():
                if self.dim == 2:
                    for i,d in enumerate(self.pos_data):
                        plt.plot(d[0], d[1], 'o', color = [0,0,1,i/50])
                    for i,d in enumerate(self.obj_data):
                        plt.plot(d[0], d[1], 'o', markersize = 2.5, color = [0,1,0,i/50])
                    plt.plot(self.goal[0], self.goal[1], 'rx')
                    if mode == 'human':
                        plt.show()
                        plt.clf()
                    elif mode == 'image':
                        fullpath = os.path.join(save_path, 'pic')
                        if not os.path.exists(fullpath):
                            os.makedirs(fullpath)
                        fullpath = os.path.join(save_path, 'pic/pic'+str(self.save_num)+'.png')
                        plt.savefig(fullpath)
                        print('picture saved in ', fullpath)
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
                    elif mode == 'image':
                        fullpath = os.path.join(save_path, 'pic')
                        if not os.path.exists(fullpath):
                            os.makedirs(fullpath)
                        fullpath = os.path.join(save_path, 'pic/pic'+str(self.save_num)+'.png')
                        plt.savefig(fullpath)
                        self.save_num += 1
                        plt.clf()

    def _get_obs(self):
        if self.use_her:
            return {
                'observation': self.pos,
                'achieved_goal': self.obj,
                'desired_goal': self.goal
            }
        else:
            return np.concatenate((self.pos,self.obj,self.goal)) 
    
    def _get_done(self):
        if self.use_her:
            self.done = (self.num_step >= self._max_episode_steps)
        else: 
            self.done = (self.num_step >= self._max_episode_steps) or (self.d_o2g < self.error)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return np.linalg.norm(achieved_goal-desired_goal) < self.error


    def ezpolicy(self, obs):
        pos = obs[:self.dim]
        obj = obs[self.dim:self.dim*2]
        goal = obs[self.dim*2:]
        if_reach = np.linalg.norm(pos - obj)< self.error
        if not if_reach:
            act =  np.append((obj - pos),1)
        else:
            act = np.append((goal - pos),1)
        if self.mode == 'dynamic':
            act = np.append(act, pos - obj)
        return act

register(
    id='NaivePickAndPlace-v0',
    entry_point='gym_naive.naive_pac:NaivePickAndPlace',
)

if __name__ == '__main__':
    env = NaivePickAndPlace(use_grasp = False)
    obs = env.reset()
    rew_data = []
    for i in range(50):
        act = env.ezpolicy(obs)
        obs, reward, done, info = env.step(act)
        env.render(mode = 'human')
        rew_data.append(reward)
        print('[obs, reward, done]', obs, reward, done)
    plt.plot(np.arange(50), rew_data)
    plt.show()