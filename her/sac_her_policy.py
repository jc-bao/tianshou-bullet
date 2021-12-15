from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch
from gym.spaces import Box, Discrete, MultiBinary, MultiDiscrete
from numba import njit
from torch import nn

from tianshou.exploration import BaseNoise
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.policy import SACPolicy, BasePolicy
from tianshou.policy.base import _nstep_return

class SACHERPolicy(SACPolicy):
    '''
    Why redesign process_fn? because the original one cannot change HER rew.
    '''
    def __init__(
        self,
        actor: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1: torch.nn.Module,
        critic1_optim: torch.optim.Optimizer,
        critic2: torch.nn.Module,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        reward_normalization: bool = False,
        estimation_step: int = 1,
        exploration_noise: Optional[BaseNoise] = None,
        deterministic_eval: bool = True,
        dict_observation_space = None,
        reward_fn = None, 
        future_k: float = 4,
        max_episode_length = 50,
        strategy = 'offline',
        **kwargs: Any,
    ) -> None:
        super().__init__(
            actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
            tau, gamma, alpha, reward_normalization, estimation_step, exploration_noise, deterministic_eval, **kwargs
        )
        self.future_k = future_k
        self.strategy = strategy
        self.future_p = 1 - (1. / (1 + future_k))
        self.reward_fn = reward_fn
        self.max_episode_length = max_episode_length
        # get index information of observation
        self.dict_observation_space = dict_observation_space # used for unflatten Note: gym flatten function is slow
        current_idx = 0
        self.index_range = {}
        for (key,s) in dict_observation_space.spaces.items():
            self.index_range[key] = np.arange(current_idx, current_idx+s.shape[0])
            current_idx += s.shape[0]

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        if self.strategy == 'offline':
            return super(SACHERPolicy, self).process_fn(batch, buffer, indices)
        assert not self._rew_norm, \
            "Reward normalization in computing n-step returns is unsupported now."
        end_flag = buffer.done.copy()
        end_flag[buffer.unfinished_index()] = True # consider unfinished case: remove it
        # step0: get all index needed
        bsz = len(indices) # get indice of sampled transitions
        indices = [indices] # turn to list, prepare for expand next state [1,3]
        for _ in range(self._n_step - 1):
            indices.append(buffer.next(indices[-1])) # append next state index [[1,3][2,4]]
        indices = np.stack(indices) # ?
        terminal = indices[-1] # next state
        # step1: sample new goal
        # if hasattr(buffer, 'buffer_num'):
        #     buffer_size = buffer.buffers[0].maxsize
        # else:
        #     buffer_size = buffer.maxsize
        # new_goal = []
        # for idx in terminal:
        #     if np.random.random() > self.future_p: # not change
        #         obs_next = buffer.obs_next[idx]
        #         new_goal.append(obs_next[self.index_range['desired_goal']])
        #     else:
        #         buffer_idx = int(idx/buffer_size)
        #         current_buffer_done = end_flag[idx:min((buffer_idx+1)*buffer_size,self.max_episode_length+idx)]
        #         next_done_distance = next((i for i, x in enumerate(current_buffer_done) if x), None) or 0
        #         new_goal_idx = int(np.random.random()*(next_done_distance+1))+idx
        #         obs_next = buffer.obs_next[new_goal_idx]
        #         new_goal.append(obs_next[self.index_range['achieved_goal']])
        # new_goal = np.array(new_goal)
        # step2: calculate Q
        batch = buffer[terminal]  # batch.obs: s_{t+n}
        new_goal = batch.obs_next[:,self.index_range['desired_goal']]
        for i in range(bsz):
            if np.random.random()<self.future_p:
                goals = batch.info.achieved_goal[i]
                try: # make sure the goal exists
                    new_goal[i] = goals[int(np.random.random()*len(goals))]
                except:
                    pass
        # relabel: change batch's obs, obs_next, reward
        batch.obs[:,self.index_range['desired_goal']] = new_goal
        batch.obs_next[:,self.index_range['desired_goal']] = new_goal
        batch.rew = self.reward_fn(batch.obs_next[:,self.index_range['achieved_goal']], new_goal, None)
        with torch.no_grad():
            obs_next_result = self(batch, input='obs_next') # [CHANGE] get action from policy 
            a_ = obs_next_result.act
            target_q_torch = torch.min( # [CHANGE] batch obs next
                self.critic1_old(batch.obs_next, a_),
                self.critic2_old(batch.obs_next, a_),
            ) - self._alpha * obs_next_result.log_prob
        target_q = to_numpy(target_q_torch.reshape(bsz, -1))
        target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
        # step3: calculate N step return
        gamma_buffer = np.ones(self._n_step + 1)
        for i in range(1, self._n_step + 1):
            gamma_buffer[i] = gamma_buffer[i - 1] * self._gamma
        target_shape = target_q.shape
        bsz = target_shape[0]
        # change target_q to 2d array
        target_q = target_q.reshape(bsz, -1)
        returns = np.zeros(target_q.shape) # n_step returrn
        gammas = np.full(indices[0].shape, self._n_step)
        for n in range(self._n_step - 1, -1, -1):
            now = indices[n]
            gammas[end_flag[now] > 0] = n + 1
            returns[end_flag[now] > 0] = 0.0
            new_rew = []
            old_obs_next = buffer.obs_next[now]
            new_rew.append(self.reward_fn(old_obs_next[:,self.index_range['achieved_goal']], new_goal, None))
            returns = np.array(new_rew).reshape(bsz, 1) + self._gamma * returns # [CHANGE] rew
        target_q = target_q * gamma_buffer[gammas].reshape(bsz, 1) + returns
        target_q = target_q.reshape(target_shape)
        # return values
        batch.returns = to_torch_as(target_q, target_q_torch)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)
        return batch