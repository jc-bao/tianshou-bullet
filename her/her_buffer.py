from typing import Any, List, Optional, Tuple, Union, Sequence

import numpy as np
from numpy.random import triangular
import torch

from tianshou.data import (
    Batch,
    ReplayBuffer,
    ReplayBufferManager,
)

class HERReplayBuffer(ReplayBuffer):
    """ Implementation of Prioritized Experience Replay. arXiv:1707.01495.
    Use future strategy to replay
    :param float k: ratio to be replaced by hindsight
    :param int achieved_goal_index: start index of achieved_goal
    :param int desired_goal_index: start index of desired_goal
    """
    
    def __init__(
        self,
        size: int,
        k: float,
        reward_fn,
        achieved_goal_index: int,
        desired_goal_index: int,
        max_reward: int,
        **kwargs: Any
    ) -> None:
        ReplayBuffer.__init__(self, size, **kwargs)
        self.future_p = 1 - (1. / (1 + k))
        self.reward_fn = reward_fn
        self.desired_goal_index = desired_goal_index
        self.achieved_goal_index = achieved_goal_index
        self.max_reward = max_reward

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        """Get a hindsight sample from buffer with size = batch_size.

        Return all the data in the buffer if batch_size is 0.

        :return: Sample data and its corresponding index inside the buffer.
        """
        # print('=========sampling....=========')
        indices = self.sample_indices(batch_size)
        transitions = self[indices]
        # index of transitions to be replaced
        replace_idx = np.random.choice(batch_size, int(batch_size*self.future_p), replace=False)
        replace_indices = indices[replace_idx]
        # get future goal
        future_length = transitions[replace_idx].info.future_length
        future_goal_indices = np.clip(np.random.randint(low = replace_indices, high = replace_indices + future_length + 1), 0, self.maxsize)
        # replace goal and reward
        dones = np.logical_or(np.equal(transitions.rew[replace_idx],self.max_reward), transitions.done[replace_idx]).reshape(-1,1) # Attention! we use reward to judge if done
        obs_old = transitions.obs[replace_idx]
        goal_future = self.obs[future_goal_indices][:, self.achieved_goal_index:self.desired_goal_index]*(1-dones) + obs_old[:, self.desired_goal_index:]*dones # if done, not replace
        obs_new = np.concatenate((obs_old[:, :self.desired_goal_index], goal_future), axis = 1)
        transitions.obs[replace_idx] = obs_new
        obs_next_old = transitions.obs_next[replace_idx]
        goal_next_future = self.obs[future_goal_indices][:, self.achieved_goal_index:self.desired_goal_index]*(1-dones) + obs_next_old[:, self.desired_goal_index:]*dones # use this to aviod conflict in done
        obs_next_new = np.concatenate((obs_next_old[:, :self.desired_goal_index], goal_next_future), axis = 1)
        transitions.obs_next[replace_idx] = obs_next_new
        for i, idx in enumerate(replace_idx):
            transitions.rew[idx] = self.reward_fn(obs_next_new[i, self.achieved_goal_index:self.desired_goal_index], obs_next_new[i, self.desired_goal_index:], None)
        return transitions, indices

class HERVectorReplayBuffer(ReplayBufferManager):

    def __init__(self, total_size: int, buffer_num: int, **kwargs: Any) -> None:
        assert buffer_num > 0
        size = int(np.ceil(total_size / buffer_num))
        buffer_list = [
            HERReplayBuffer(size, **kwargs) for _ in range(buffer_num)
        ]
        super().__init__(buffer_list)