import gym
from gym import spaces
import numpy as np
import logging
from gym.envs.registration import register

class NaivePickAndPlace(gym.Env):
    def __init__(self, reward_type='dense'):
        self.reward_type = reward_type
        self._max_episode_steps = 50
        self.space = spaces.Box(low=np.array([0,0]), high=np.array([1,1]))
        self.observation_space = self.space
        self.action_space = self.space
        self.reset()

    def step(self, action):
        self.num_step += 1
        action = np.clip(action, self.space.low, self.space.high)
        self.pos += np.clip(action * 0.10, self.space.low, self.space.high)
        obs = self.pos
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
        return self.pos

# logger = logging.getLogger(__name__)
register(
    id='NaiveReach-v0',
    entry_point='test_env.naive_reach:NaiveReach',
)