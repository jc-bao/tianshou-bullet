import gym
from gym import spaces
import numpy as np

class CoinFlip(gym.Env):
    def __init__(self, config):
        self.n = config['n']
        self.reset()
        self.achieved_goal_index = 0
        self.desired_goal_index = config['n']
        self.action_space = spaces.Discrete(config['n'])
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Discrete(config['n']),
            achieved_goal=spaces.Discrete(config['n']),
            observation=spaces.Discrete(0),
        ))

    def step(self, act):
        self.num_steps += 1
        self.coin[act] = not self.coin[act]
        obs = {
            'observation': [],
            'achieved_goal': self.coin,
            'desired_goal': self.goal
        }
        rew = float(self.coin==self.goal)
        done = rew or (self.num_steps==self.n)
        info ={
            'is_success': rew,
            'future_length': self.n - self.num_steps
        }
        return obs, rew, done, info
    
    def reset(self):
        self.coin = [np.random.uniform() < 0.5 for _ in range(self.n)]
        self.goal = [np.random.uniform() < 0.5 for _ in range(self.n)]
        self.num_steps = 0
        return {
            'observation': [],
            'achieved_goal': self.coin,
            'desired_goal': self.goal
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        return float((achieved_goal==desired_goal).all())
        
gym.register(
    id='CoinFlip-v0',
    entry_point='test_env.coin_flip:CoinFlip',
)