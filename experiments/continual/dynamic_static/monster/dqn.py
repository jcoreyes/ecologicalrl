"""
Run DQN on grid world.
"""
import math
from os.path import join

import gym
import copy

from gym_minigrid.envs.lava import LavaEnv
from gym_minigrid.envs.monsters import MonstersEnv
from gym_minigrid.envs.tools import ToolsEnv
from rlkit.core.logging import get_repo_dir
from rlkit.samplers.data_collector.path_collector import LifetimeMdpPathCollector, MdpPathCollectorConfig
from rlkit.torch.dqn.double_dqn import DoubleDQNTrainer
from rlkit.torch.sac.policies import SoftmaxQPolicy
from torch import nn as nn
import rlkit.util.hyperparameter as hyp
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy, EpsilonGreedySchedule, EpsilonGreedyDecay
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.dqn.dqn import DQNTrainer
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger, run_experiment
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm, TorchLifetimeRLAlgorithm

# from variants.dqn.dqn_medium_mlp_task_partial_variant import variant as algo_variant, gen_network
from variants.dqn_lifetime.dqn_medium8_mlp_task_partial_variant import variant as algo_variant, gen_network#_num_obj as gen_network


def schedule(t):
    print(t)
    return max(1 - 5e-4 * t, 0.05)


def experiment(variant):
    from rlkit.envs.gym_minigrid.gym_minigrid import envs

    expl_env = MonstersEnv(
        **variant['env_kwargs']
    )
    eval_env = MonstersEnv(
        **variant['env_kwargs']
    )
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.n
    layer_size = variant['algo_kwargs']['layer_size']
    lifetime = variant['env_kwargs'].get('time_horizon', 0) == 0

    qf = gen_network(variant['algo_kwargs'], action_dim, layer_size)
    target_qf = gen_network(variant['algo_kwargs'], action_dim, layer_size)

    qf_criterion = nn.MSELoss()
    eval_policy = ArgmaxDiscretePolicy(qf)
    # eval_policy = SoftmaxQPolicy(qf)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        EpsilonGreedyDecay(expl_env.action_space, variant['algo_kwargs']['eps_decay_rate'], 1, 0.1),
        eval_policy,
    )
    if lifetime:
        eval_policy = expl_policy
    # expl_policy = PolicyWrappedWithExplorationStrategy(
    #     EpsilonGreedy(expl_env.action_space, 0.5),
    #     eval_policy,
    # )
    if eval_env.time_horizon == 0:
        collector_class = LifetimeMdpPathCollector if lifetime else MdpPathCollector
    else:
        collector_class = MdpPathCollectorConfig
    eval_path_collector = collector_class(
        eval_env,
        eval_policy,
        # render=True
    )
    expl_path_collector = collector_class(
        expl_env,
        expl_policy
    )
    trainer = DoubleDQNTrainer(
        qf=qf,
        target_qf=target_qf,
        qf_criterion=qf_criterion,
        **variant['algo_kwargs']['trainer_kwargs']
    )
    replay_buffer = EnvReplayBuffer(
        variant['algo_kwargs']['replay_buffer_size'],
        expl_env
    )
    algo_class = TorchLifetimeRLAlgorithm if lifetime else TorchBatchRLAlgorithm
    algorithm = algo_class(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    """
    NOTE: Things to check for running exps:
    1. Mode (local vs ec2)
    2. algo_variant, env_variant, env_search_space
    3. use_gpu 
    """
    exp_prefix = 'monster-food'
    n_seeds = 1
    mode = 'local'
    use_gpu = False

    env_variant = dict(
        grid_size=10,
        agent_start_pos=None,
        health_cap=1000,
        gen_resources=True,
        fully_observed=False,
        task='make food',
        make_rtype='sparse',
        fixed_reset=False,
        only_partial_obs=True,
        init_resources={
            'food': 2,
            'monster': 2
        },
        resource_prob={
            'food': 0.0,
            'monster': 0.0
        },
        lifespans={
            'monster': 20,
        },
        replenish_low_resources={
            'food': 2,
            'monster': 2
        },
        end_on_task_completion=False,
        fixed_expected_resources=True,
        time_horizon=0,
        agent_view_size=5
    )
    env_search_space = copy.deepcopy(env_variant)
    env_search_space = {k: [v] for k, v in env_search_space.items()}
    env_search_space.update(
        monster_eps=[
            0, 0.2, 0.4
        ],
        # env shaping
        monster_attack_dist=[
            0, 1
        ],
        monster_penalty=[-100],
        lifespans=[
            {'monster': 10},
            {'monster': 20},
            {'monster': 30},
        ],
        # # resource conditions
        # init_resources=[
        #     {'food': 1, 'monster': 1},
        #     {'food': 2, 'monster': 2},
        # ],
        resource_prob=[
            {'food': 0.0, 'monster': 0.0},
            # {'food': 0.1, 'monster': 0.0},
            {'food': 0.2, 'monster': 0.0}
        ],
        # reward shaping
        make_rtype=[
            'sparse', 'dense-fixed', 'waypoint', #'one-time',
        ],
        # reset / reset free
        time_horizon=[
             200
        ]
    )

    algo_variant = dict(
        algorithm="DQN",
        version="monster - food",
        layer_size=16,
        replay_buffer_size=int(5E5),
        eps_decay_rate=1e-5,
        algorithm_kwargs=dict(
            num_epochs=2500,
            num_eval_steps_per_epoch=6000,
            num_trains_per_train_loop=500,
            num_expl_steps_per_train_loop=500,
            min_num_steps_before_training=200,
            max_path_length=math.inf,
            batch_size=64,
            validation_envs_pkl=join(get_repo_dir(), 'experiments/continual/dynamic_static/monster/validation_envs/dynamic_static_validation_envs_2020_02_05_02_06_18.pkl'),
            validation_rollout_length=200,
            validation_period=10,
            # store visit count array for heat map
            viz_maps=True,
            viz_gap=100
        ),
        trainer_kwargs=dict(
            discount=0.99,
            learning_rate=1E-4,
            grad_clip_val=5
        ),
        inventory_network_kwargs=dict(
            # shelf: 8 x 8
            input_size=64,
            output_size=16,
            hidden_sizes=[16, 16],
        ),
        full_img_network_kwargs=dict(
            # 5 x 5 x 8
            input_size=200,
            output_size=32,
            hidden_sizes=[64, 64]
        ),
        num_obj_network_kwargs=dict(
            # num_objs: 8
            input_size=8,
            output_size=8,
            hidden_sizes=[8]
        )
    )
    algo_search_space = copy.deepcopy(algo_variant)
    algo_search_space = {k: [v] for k, v in algo_search_space.items()}
    algo_search_space.update(
        # insert sweep params here
    )

    env_sweeper = hyp.DeterministicHyperparameterSweeper(
        env_search_space, default_parameters=env_variant,
    )
    algo_sweeper = hyp.DeterministicHyperparameterSweeper(
        algo_search_space, default_parameters=algo_variant,
    )

    for exp_id, env_vari in enumerate(env_sweeper.iterate_hyperparameters()):
        for algo_vari in algo_sweeper.iterate_hyperparameters():
            variant = {'algo_kwargs': algo_vari, 'env_kwargs': env_vari}
            for _ in range(n_seeds):
                run_experiment(
                    experiment,
                    exp_prefix=exp_prefix,
                    mode=mode,
                    variant=variant,
                    use_gpu=use_gpu,
                    region='us-east-2',
                    num_exps_per_instance=3,
                    snapshot_mode='gap',
                    snapshot_gap=10,
                    instance_type='c4.large',
                    spot_price=0.07
                )
