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
from tianshou.policy import SACPolicy
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

import gym_naive
from her.offpolicy import offpolicy_trainer # custom off policy trainer, add success log
import time
from functools import partial
from her.sac_her_policy import SACHERPolicy
from her.her_collector import HERCollector

if __name__ == '__main__':
    '''
    load param
    '''
    with open("config/pnp_dai.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    '''
    make env
    '''
    def make_env():
        # return gym.wrappers.FlattenObservation(gym.make(config['env'], config = config))
        return gym.wrappers.FlattenObservation(gym.make(config['env']))
    def make_record_env(i):
        # return gym.wrappers.FlattenObservation(gym.make(config['env'], config = config))
        return gym.wrappers.RecordVideo(gym.wrappers.FlattenObservation(gym.make(config['env'])), video_folder = 'log/video/'+'bar'+str(i))
    # env = gym.make(config['env'], config = config)
    env = gym.make(config['env'])
    observation_space = env.observation_space
    env = gym.wrappers.FlattenObservation(env)
    obs = env.reset()
    state_shape = len(obs)
    action_shape = env.action_space.shape or env.action_space.n
    train_envs = SubprocVectorEnv(
        [make_env for _ in range(config['training_num'])],
        norm_obs = config['norm_obs']
    )
    test_envs = SubprocVectorEnv(
        [partial(make_record_env, i) for i in range(config['test_num'])], 
        norm_obs = config['norm_obs'],
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
    # actor
    net_a = Net(state_shape, hidden_sizes=config['hidden_sizes'], device=config['device'])
    actor = ActorProb(
        net_a,
        action_shape,
        max_action=env.action_space.high[0],
        device=config['device'],
        unbounded=True,
        conditioned_sigma=True
    ).to(config['device'])
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config['actor_lr'])
    # critic
    net_c1 = Net(
        state_shape,
        action_shape,
        hidden_sizes=config['hidden_sizes'],
        concat=True,
        device=config['device']
    )
    net_c2 = Net(
        state_shape,
        action_shape,
        hidden_sizes=config['hidden_sizes'],
        concat=True,
        device=config['device']
    )
    critic1 = Critic(net_c1, device=config['device']).to(config['device'])
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=config['critic_lr'])
    critic2 = Critic(net_c2, device=config['device']).to(config['device'])
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=config['critic_lr'])
    # auto alpha
    if config['auto_alpha']:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=config['device'])
        alpha_optim = torch.optim.Adam([log_alpha], lr=config['alpha_lr'])
        config['alpha'] = (target_entropy, log_alpha, alpha_optim)

    '''
    set up policy
    '''
    policy = SACHERPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=config['tau'],
        gamma=config['gamma'],
        alpha=config['alpha'],
        estimation_step=config['estimation_step'],
        action_space=env.action_space,
        reward_normalization = False,
        dict_observation_space = observation_space,
        reward_fn = env.compute_reward, 
        future_k = config['replay_k'],
    )
    # load policy
    if config['resume_path']:
        policy.load_state_dict(torch.load(config['resume_path'], map_location=config['device']))
        print("Loaded agent from: ", config['resume_path'])

    '''
    set up collector
    '''
    if config['training_num'] > 1:
        buffer = VectorReplayBuffer(config['buffer_size'], len(train_envs))
    else:
        buffer = ReplayBuffer(config['buffer_size'])
    train_collector = HERCollector(policy, train_envs, buffer, exploration_noise=True, observation_space = observation_space, reward_fn = env.compute_reward, k = config['replay_k'], strategy=config['strategy'])
    test_collector = Collector(policy, test_envs)
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
        # save render data
        '''
        obs = env.reset()
        done = False
        while not done:
            obs = np.array(list(obs.values())).flatten()
            data = Batch(
                obs=[obs], act={}, rew={}, done={}, obs_next={}, info={}, policy={}
            )
            with torch.no_grad():  # faster than retain_grad version
                result = policy(data, None)
            action_remap = policy.map_action(result.act)
            obs, rew, done, info = env.step(action_remap[0].detach().cpu().numpy())
            env.render(mode = 'tensorboard', writer = writer)
        '''
        
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