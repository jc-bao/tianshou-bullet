from tianshou.data import ReplayBuffer

class HERReplayBuffer(ReplayBuffer):
    """ Implementation of Prioritized Experience Replay. arXiv:1707.01495.
    
    :param float k: replay with k random states
    """
