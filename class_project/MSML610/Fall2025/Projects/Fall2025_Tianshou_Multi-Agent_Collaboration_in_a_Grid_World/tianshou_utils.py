"""
tianshou_utils.py

This file contains the classes that implement the grid world environment, 
and the functions to run the training loop.
"""
import time
import os
import numpy as np
import argparse
from gymnasium import spaces
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import Collector, VectorReplayBuffer, Batch
from tianshou.env import DummyVectorEnv
from tianshou.policy import (
    BasePolicy,
    DQNPolicy,
    MultiAgentPolicyManager,
    RandomPolicy,
)
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net
from tianshou.utils import TensorboardLogger
from tianshou.utils import BaseLogger  
        
def get_parser() -> argparse.ArgumentParser:
    """
    Sets parameters that will be used for training.
    
    :return: a parser that parses arguments defined in this function.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626) # Seed
    parser.add_argument('--eps-test', type=float, default=0.05) # Test epsilon
    parser.add_argument('--eps-train', type=float, default=0.95) # Training epsilon: controls initial exploration
    parser.add_argument('--buffer-size', type=int, default=100000) 
    parser.add_argument('--lr', type=float, default=1e-4) # Learning rate
    parser.add_argument(
        '--gamma', type=float, default=0.95) # Gamma controls future reward decay
    parser.add_argument('--n-step', type=int, default=3) # Number of steps in between (bootstrapped) returns
    parser.add_argument('--target-update-freq', type=int, default=320) 
    parser.add_argument('--epoch', type=int, default=80) #Epochs
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument(
        '--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128]
    )
    parser.add_argument('--training-num', type=int, default=10) # Number of training environments to create
    parser.add_argument('--test-num', type=int, default=10) # Number of test environments to create
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='no training, '
        'watch the play of pre-trained models'
    )
    parser.add_argument(
        '--num-agents',
        type=int,
        default=2, 
        help='number of agents'
    )
    parser.add_argument(
        '--resume-path',
        type=str,
        default='',
        help='the path of agent pth file '
        'for resuming from a pre-trained agent'
    )
    parser.add_argument(
        '--opponent-path',
        type=str,
        default='',
        help='the path of opponent agent pth file '
        'for resuming from a pre-trained agent'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument(
        '--opponent-sync-interval', type=int, default=1,
        help='epochs between copying learner weights to opponent')
    return parser
    
def get_agents(get_env, args, network, agent_learn = None, optim = None):
    """
    Creates the agents and policy manager for the Grid World.
    
    :param get_env: function to generate the training and test environments. Function format is required for vectorized environment.
    :param args: the arguments from the parser.
    :param network: function to create the network used by the agent.
    :param agent_learn: the learning agent's policy. By default, uses DQN.
    :param optim: the optimizer to use. By default, uses Adam.
    
    :return: Multi Agent Policy Manager with each agent's policies, optimizer, and list of agents by name
    """
    env = get_env()
    observation_space = env.observation_space
    if isinstance(observation_space, spaces.Dict):
        local_shape = observation_space["local"].shape
        knowledge_dim = observation_space["knowledge"].shape[0]
    else:
        raise ValueError("GridWorld should expose a Dict observation.")
    args.action_shape = env.action_space.shape or env.action_space.n
    if agent_learn is None:
        # Create the default learning agent policy, which is Deep Q-Network (DQN).
        obs_space = env.observation_space["local"]
        local_shape = obs_space.shape
        knowledge_dim = env.observation_space["knowledge"].shape[0]
        net = network(local_shape, knowledge_dim, args.action_shape, env.env.num_coins, len(env.env.possible_agents)).to(args.device)
        
        # Create the default optimizer, which is Adam.
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        agent_learn = DQNPolicy(
            net,
            optim,
            args.gamma,
            args.n_step,
            target_update_freq=args.target_update_freq
        )
        if args.resume_path:
            agent_learn.load_state_dict(torch.load(args.resume_path))
    agents = [agent_learn for _ in range(args.num_agents)]
    
    policy = MultiAgentPolicyManager(agents, env)
    return policy, optim, env.agents

class EpochTrainRewardLogger(TensorboardLogger):
    """
    Implements a custom logger that displays the average training reward at every epoch.
    """
    def __init__(self, writer):
        """
        Subclasses the base logger, which displays an average test reward and the best test reward after every epoch.
        
        :param writer: the base logger.
        """
        super().__init__(writer)
        self._epoch_rewards = []

    def log_train_data(self, collect_result, step):
        """
        Adds all rewards to self._epoch_rewards.
        
        :param collect_result: the Collector's collection of the current step's outcome.
        :param step: the current step number.
        """
        super().log_train_data(collect_result, step)
        rews = collect_result["rews"]
        if rews.size > 0:
            self._epoch_rewards.append(rews.mean())

    def log_test_data(self, collect_result, step):
        """
        Called at end of epoch after test collection. 
        Averages and prints the average training reward for the epoch.
        Then, resets the list.
        
        :param collect_result: the Collector's collection of the current step's outcome.
        :param step: the current step number.
        """
        if self._epoch_rewards:
            epoch_mean = sum(self._epoch_rewards) / len(self._epoch_rewards)
            print(f"[Epoch] env_step={step}  avg_train_reward={epoch_mean:.3f}")
            self._epoch_rewards = []
        super().log_test_data(collect_result, step)

def train_agent(get_env, args, network, agent_learn = None, optim = None):
    """
    Implements the training loop for the agent.
    
    :param get_env: function to generate the training and test environments. Function format is required for vectorized environment.
    :param args: the arguments from the parser.
    :param network: function to create the network used by the agent.
    :param agent_learn: the learning agent's policy. By default, uses DQN.
    :param optim: the optimizer to use. By default, uses Adam.
    
    :return: the result of training.
    """
    # Create (pseudo-)vectorized training and test environments
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # Create agents, policy, and optimizer
    policy, optim, agents = get_agents(
        get_env, args, network, agent_learn=agent_learn, optim=optim
    )

    # Create the Collector, which is responsible for accumulating the results of every environment for the network to learn.
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    train_collector.collect(n_step=args.batch_size * args.training_num)

    # Create the logger
    log_path = os.path.join(args.logdir, 'grid_world', 'dqn')
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = EpochTrainRewardLogger(writer)

    # Callback functions used during training
    def train_fn(epoch, env_step):
        """
        Called at the start of every training episode (step). 
        Sets the training epsilon for each agent, which linearly decays to 0.1.
        
        :param env_step: the current step of the environment
        """
        eps_start, eps_end = args.eps_train, 0.1
        decay_steps = args.step_per_epoch * args.epoch
        frac = min(env_step / decay_steps, 1.0)
        eps = eps_start + frac * (eps_end - eps_start)
        
        for agent_name in agents:
            policy.policies[agent_name].set_eps(eps)

    def test_fn(epoch, env_step):
        """
        Called at the start of every test episode (step).
        Sets the test epsilon for each agent.
        
        :param env_step: the current step of the environment
        """
        for agent_name in agents:
            policy.policies[agent_name].set_eps(args.eps_test)

    def reward_metric(rews):
        """
        Returns the average reward for all test environments in the epoch.
        """
        return rews[:, 0].mean()

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        update_per_step=args.update_per_step,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric
    )

    return result, policy.policies[agents[0]]

def watch(get_env, args, network, agent_learn = None):
    """
    Displays a rendered episode of training.
    
    :param get_env: function to generate the training and test environments. Function format is required for vectorized environment.
    :param args: the arguments from the parser.
    :param network: function to create the network used by the agent.
    :param agent_learn: the learning agent's policy. By default, uses DQN.
    :param optim: the optimizer to use. By default, uses Adam.
    """
    env = get_env(render_mode="human")
    env = DummyVectorEnv([lambda: env])
    policy, optim, agents = get_agents(
        get_env, args, network, agent_learn=agent_learn
    )
    policy.eval()
    policy.policies[agents[0]].set_eps(args.eps_test)
    collector = Collector(policy, env, exploration_noise=True)
    result = collector.collect(n_episode=1, render=args.render)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, 0].mean()}, length: {lens.mean()}")

def get_centralized_policy(get_env, args, network, agent_learn=None, optim=None):
    """
    Creates a single policy to manage agents for the Grid World.
    
    :param get_env: function to generate the training and test environments. Function format is required for vectorized environment.
    :param args: the arguments from the parser.
    :param network: function to create the network used by the agent.
    :param agent_learn: the learning agent's policy. By default, uses DQN.
    :param optim: the optimizer to use. By default, uses Adam.
    
    :return: a single policy and an optimizer
    """
    env = get_env()
    obs_space = env.observation_space
    action_dim = env.action_space.n
    local_shape = obs_space["local"].shape
    knowledge_dim = obs_space["knowledge"].shape[0]
    if agent_learn is None:
        net = network(local_shape, knowledge_dim, action_dim, env.num_coins, args.num_agents).to(args.device)
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        agent_learn = DQNPolicy(net, optim, args.gamma, args.n_step,
                                target_update_freq=args.target_update_freq)
    return agent_learn, optim

def train_centralized_policy(get_env, args, network, agent_learn = None, optim = None):
    """
    Implements the training loop for the agent.
    Separate function, because setting a "centralized" parameter in the other functions breaks them.
    Implementation is otherwise nearly identical, except that there is a single policy.
    
    :param get_env: function to generate the training and test environments. Function format is required for vectorized environment.
    :param args: the arguments from the parser.
    :param network: function to create the network used by the agent.
    :param agent_learn: the learning agent's policy. By default, uses DQN.
    :param optim: the optimizer to use. By default, uses Adam.
    
    :return: the result of training.
    """
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    
    policy, optim = get_centralized_policy(
        get_env, args, network, agent_learn=agent_learn, optim=optim
    )
    
    train_collector = Collector(policy, train_envs,
                                VectorReplayBuffer(args.buffer_size, len(train_envs)),
                                exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    train_collector.collect(n_step=args.batch_size * args.training_num)

    def train_fn(epoch, env_step):
        eps_start, eps_end = args.eps_train, 0.1
        decay_steps = args.step_per_epoch * args.epoch
        frac = min(env_step / decay_steps, 1.0)
        policy.set_eps(eps_start + frac * (eps_end - eps_start))

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)
    
    log_path = os.path.join(args.logdir, 'grid_world', 'dqn')
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = EpochTrainRewardLogger(writer)
    
    result = offpolicy_trainer(
        policy, train_collector, test_collector,
        args.epoch, args.step_per_epoch, args.step_per_collect,
        args.test_num, args.batch_size,
        train_fn=train_fn, test_fn=test_fn,
        update_per_step=args.update_per_step,
        logger=logger,
    )
    return result, policy

def watch_centralized(get_env, args, network, agent_learn=None):
    """
    Displays a rendered episode of training from the centralized policy. 
    Separate function, because setting a "centralized" parameter in the other functions breaks them.
    Implementation is otherwise nearly identical, except that there is a single policy.
    
    :param get_env: function to generate the training and test environments. Function format is required for vectorized environment.
    :param args: the arguments from the parser.
    :param network: function to create the network used by the agent.
    :param agent_learn: the learning agent's policy. By default, uses DQN.
    :param optim: the optimizer to use. By default, uses Adam.
    """
    env = DummyVectorEnv([get_env])
    policy, optim = get_centralized_policy(
        get_env, args, network, agent_learn=agent_learn
    )
    policy.eval()
    policy.set_eps(args.eps_test)
    collector = Collector(policy, env, exploration_noise=False)
    result = collector.collect(n_episode=1, render=args.render)
    time.sleep(0.05)
    print(f"Final reward: {result['rews'].mean()}, length: {result['lens'].mean()}")