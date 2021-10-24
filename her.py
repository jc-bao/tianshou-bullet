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
        **kwargs: Any
    ) -> None:
        ReplayBuffer.__init__(self, size, **kwargs)
        self.future_p = 1 - (1. / (1 + k))
        self.reward_fn = reward_fn
        self.desired_goal_index = desired_goal_index
        self.achieved_goal_index = achieved_goal_index

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        """Get a hindsight sample from buffer with size = batch_size.

        Return all the data in the buffer if batch_size is 0.

        :return: Sample data and its corresponding index inside the buffer.
        """
        indices = self.sample_indices(batch_size)
        transitions = self[indices]
        # index of transitions to be replaced
        replace_idx = np.random.choice(batch_size, int(batch_size*self.future_p))
        replace_indices = indices[replace_idx]···
        # get future goal
        future_goal_indices = replace_indices + (np.random.uniform(size=len(replace_idx)) * transitions[replace_idx].info.future_length).astype(int) + 1
        # replace goal and reward
        transitions[replace_idx].obs[:, self.desired_goal_index:] = self[future_goal_indices].obs[:, self.achieved_goal_index:self.desired_goal_index]
        transitions[replace_idx].reward = [self.reward_fn(transitions[i].obs[self.achieved_goal_index:self.desired_goal_index], transitions[i].obs[self.desired_goal_index:], transitions[i].info) for i in replace_idx]
        return transitions, indices

class HERVectorReplayBuffer(ReplayBufferManager):

    def __init__(self, total_size: int, buffer_num: int, **kwargs: Any) -> None:
        assert buffer_num > 0
        size = int(np.ceil(total_size / buffer_num))
        buffer_list = [
            HERReplayBuffer(size, **kwargs) for _ in range(buffer_num)
        ]
        super().__init__(buffer_list)