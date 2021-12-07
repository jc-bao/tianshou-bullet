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
        **kwargs: Any,
    ) -> None:
        super().__init__(
            actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
            tau, gamma, alpha, reward_normalization, estimation_step, exploration_noise, deterministic_eval, **kwargs
        )
        self.future_p = 1 - (1. / (1 + future_k))
        self.reward_fn = reward_fn
        self.dict_observation_space = dict_observation_space # used for unflatten

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        assert not self.rew_norm, \
            "Reward normalization in computing n-step returns is unsupported now."
        rew = buffer.rew
        bsz = len(indices)
        indices = [indices]
        for _ in range(self._n_step - 1):
            indices.append(buffer.next(indices[-1])) # append final state
        indices = np.stack(indices)
        # terminal indicates buffer indexes nstep after 'indice',
        # and are truncated at the end of each episode
        terminal = indices[-1]
        with torch.no_grad():
            target_q_torch = self._target_q(buffer, terminal)  # (bsz, ?)
        target_q = to_numpy(target_q_torch.reshape(bsz, -1))
        target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
        end_flag = buffer.done.copy()
        end_flag[buffer.unfinished_index()] = True
        target_q = _nstep_return(rew, end_flag, target_q, indices, self._gamma, self._n_step)
        
        batch.returns = to_torch_as(target_q, target_q_torch)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)
        return batch