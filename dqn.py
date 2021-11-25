import argparse
import datetime
import os
import pprint
import yaml

import gym, gym_xarm
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer, Batch
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

from test_env import coin_flip
from her import HERReplayBuffer, HERVectorReplayBuffer

if __name__ == '__main__':
    '''
    load param
    '''
    with open("config/coin_flip.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    '''
    make env
    '''
    env = gym.make(config['env'], config = config)
    obs = env.reset()
    if config['use_her']:
        obs = np.concatenate(list(obs.values()))
    state_shape = len(obs)
    action_shape = env.action_space.shape or env.action_space.n
    train_envs = SubprocVectorEnv(
        [lambda: gym.make(config['env'], config = config) for _ in range(config['training_num'])],
        norm_obs = not config['use_her'] 
    )
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(config['env'], config = config) for _ in range(config['test_num'])],
        norm_obs = not config['use_her'] ,
        obs_rms=train_envs.obs_rms,
        update_obs_rms = False
    )
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    train_envs.seed(config['seed'])
    test_envs.seed(config['seed'])

    '''
    build and init network
    '''
    if not (torch.cuda.is_available()):
        config['device'] = 'cpu'
    net = Net(state_shape = state_shape, action_shape=action_shape, hidden_sizes=config['hidden_sizes'], device=config['device'])
    optim = torch.optim.Adam(net.parameters(), lr=config['critic_lr'])
    # define policy
    '''
    set up policy
    '''
    policy = DQNPolicy(
        net,
        optim,
        config['gamma'],
        config['estimation_step'],
        target_update_freq=config['target_update_freq']
    )
    # load policy
    if config['resume_path']:
        policy.load_state_dict(torch.load(config['resume_path'], map_location=config['device']))
        print("Loaded agent from: ", config['resume_path'])


    '''
    set up collector
    '''
    if config['use_her']:
        # Note: need get index of different type of obervations as indicator after flatten
        obs = env.reset()
        achieved_goal_index = len(obs['observation'])
        desired_goal_index = len(obs['observation']) + len(obs['achieved_goal'])
        if config['training_num'] > 1:
            buffer = HERVectorReplayBuffer(total_size = config['buffer_size'], buffer_num = len(train_envs), k = config['replay_k'], reward_fn = env.compute_reward, achieved_goal_index=achieved_goal_index, desired_goal_index=desired_goal_index)
        else:
            buffer = HERReplayBuffer(config['buffer_size'], k = config['replay_k'], reward_fn = env.compute_reward, achieved_goal_index=achieved_goal_index, desired_goal_index=desired_goal_index)
    else:
        if config['training_num'] > 1:
            buffer = VectorReplayBuffer(config['buffer_size'], len(train_envs))
        else:
            buffer = ReplayBuffer(config['buffer_size'])
    def preprocess_fn(**kwargs):
        if 'rew' not in kwargs:
            return Batch(
                obs = [np.concatenate(list(o.values())) for o in kwargs['obs']]
            )
        else:
            return Batch(
                obs_next = [np.concatenate(list(o.values())) for o in kwargs['obs_next']]
            )
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True, preprocess_fn = preprocess_fn if config['use_her'] else None)
    test_collector = Collector(policy, test_envs, preprocess_fn = preprocess_fn if config['use_her'] else None)
    # warm up
    train_collector.collect(n_step=config['start_timesteps'], random=True)

    '''
    logger
    '''
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = 'seed_'+str(config['seed'])+'_'+t0+'_'+config['env']+'_sac'
    log_path = os.path.join(config['logdir'], config['env'], 'sac', log_file)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(config))
    logger = TensorboardLogger(writer, update_interval=100, train_interval=100)
    # save function
    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))
        
    '''
    trainer
    '''
    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        config['epoch'],
        config['step_per_epoch'],
        config['step_per_collect'],
        config['test_num'],
        config['batch_size'],
        save_fn=save_fn,
        logger=logger,
        update_per_step=config['update_per_step'],
        test_in_train=False
    )
    pprint.pprint(result)