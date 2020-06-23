"""
Run DQN on grid world.
"""
import math
from os.path import join

import gym
import copy

from gym_minigrid.envs.deer import DeerEnv
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

    expl_env = DeerEnv(
        **variant['env_kwargs']
    )
    eval_env = DeerEnv(
        **variant['env_kwargs']
    )
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.n
    layer_size = variant['algo_kwargs']['layer_size']
    lifetime = variant['env_kwargs'].get('time_horizon', 0) == 0
    if lifetime:
        assert eval_env.time_horizon == 0, 'cannot have time horizon for lifetime env'

    qf = gen_network(variant['algo_kwargs'], action_dim, layer_size)
    target_qf = gen_network(variant['algo_kwargs'], action_dim, layer_size)

    qf_criterion = nn.MSELoss()
    eval_policy = ArgmaxDiscretePolicy(qf)
    # eval_policy = SoftmaxQPolicy(qf)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        EpsilonGreedyDecay(expl_env.action_space, 1e-5, 1, 0.1),
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
    exp_prefix = 'tool-dqn-env-shaping-natural-curriculum-deer-intermediate'
    n_seeds = 1
    mode = 'ec2'
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
            'metal': 1,
            'wood': 1
        },
        default_lifespan=0,
        fixed_expected_resources=True,
        end_on_task_completion=False,
        time_horizon=0,
        replenish_low_resources={
            'metal': 3,
            'wood': 3,
            'deer': 3
        },
        deer_kill_easy_decay=1e-5
    )
    env_search_space = copy.deepcopy(env_variant)
    env_search_space = {k: [v] for k, v in env_search_space.items()}
    env_search_space.update(
        init_resources=[
            # 3 and 3 case with no axes is baseline
            {'metal': 3, 'wood': 3, 'axe': 0, 'deer': 3},
            {'metal': 3, 'wood': 3, 'axe': 3, 'deer': 3},
            {'metal': 6, 'wood': 6, 'axe': 6, 'deer': 3},
            {'metal': 9, 'wood': 9, 'axe': 9, 'deer': 3}
        ],
        make_rtype=['sparse', 'dense-fixed', 'one-time', 'waypoint'],
        grid_size=[10, 16],
        deer_kill_easy_prob=[0, 0.3]
    )

    algo_variant = dict(
        algorithm="DQN Lifetime",
        version="natural curriculum - deer intermediate",
        lifetime=True,
        layer_size=16,
        replay_buffer_size=int(5E5),
        algorithm_kwargs=dict(
            num_epochs=2000,
            num_eval_steps_per_epoch=6000,
            num_trains_per_train_loop=500,
            num_expl_steps_per_train_loop=500,
            min_num_steps_before_training=200,
            max_path_length=math.inf,
            batch_size=64,
            validation_envs_pkl=join(get_repo_dir(), 'examples/continual/env_shaping/natural_curriculum/deer/validation_envs/dynamic_static_validation_envs_2019_11_03_15_43_40.pkl'),
            validation_rollout_length=1000,
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
                    instance_type='c5.large',
                    spot_price=0.08
                )
