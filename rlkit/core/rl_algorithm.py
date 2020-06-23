import abc
from collections import OrderedDict
import pickle

from gym.spaces import Discrete
from gym_minigrid.minigrid_absolute import TYPE_TO_CLASS_ABS
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.samplers.rollout_functions import rollout
import torch

import numpy as np
import matplotlib.pyplot as plt
import gtimer as gt

from rlkit.core import logger, eval_util
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import DataCollector


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times


class BaseRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: DataCollector,
            evaluation_data_collector: DataCollector,
            replay_buffer: ReplayBuffer,
            viz_maps=False,
            viz_gap=50,
            #  : validation tasks for continual proj
            validation_envs_pkl=None,
            validation_period=10,
            validation_rollout_length=100
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        self.viz_maps = viz_maps
        self.viz_gap = viz_gap
        if validation_envs_pkl is not None:
            self.validation = True
            self.validation_envs_pkl = validation_envs_pkl
            self.validation_period = validation_period
            self.validation_rollout_length = validation_rollout_length
        else:
            self.validation = False
        self._start_epoch = 0

        self.post_epoch_funcs = []

    def train(self, start_epoch=0, **kwargs):
        self._start_epoch = start_epoch
        self._train(**kwargs)

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    def marginal_entropy(self, obs_count, step_count):
        entropy = 0
        for s, count in obs_count.items():
            prob = count / step_count
            entropy -= prob * np.log(prob)
        return entropy

    def transition_entropy(self, transition_count, step_count):
        cond_entropy = 0
        state_entropies = {}
        for s, next_s_dict in transition_count.items():
            total_next_s = sum(next_s_dict.values())
            state_entropy = 0
            for next_s, count in next_s_dict.items():
                cond_prob = count / total_next_s
                joint_prob = count / step_count
                cond_entropy -= joint_prob * np.log(cond_prob)
                state_entropy -= cond_prob * np.log(cond_prob)
            state_entropies[s] = state_entropy
        return cond_entropy, state_entropies

    def tv_uniform(self, obs_count, step_count):
        num_states = len(obs_count)
        obs_count_arr = np.array(list(obs_count.values()))
        obs_prob_arr = obs_count_arr / obs_count_arr.sum()
        tv_dist = 0.5 * np.abs(obs_prob_arr - np.ones_like(obs_prob_arr) / num_states).sum()
        return tv_dist

    def kl_uniform(self, obs_count, step_count):
        """ Returns KL(obs_dist || U) """
        return np.log(len(obs_count)) - self.marginal_entropy(obs_count, step_count)

    def _end_epoch(self, epoch, incl_expl=True, minigrid=True):
        snapshot = self._get_snapshot()
        eval_env = snapshot['evaluation/env']
        if self.viz_maps and epoch % self.viz_gap == 0:
            logger.save_viz(epoch, snapshot, self.eval_env.visit_count)
        if self.validation and epoch % self.validation_period == 0:
            if minigrid:
                stats = self.validate(snapshot)
                logger.save_stats(epoch, stats, final=False)
            else:
                stats = self.get_validation_returns(snapshot)
                logger.save_stats(epoch, stats, final=True)
            # env stats (not validation)
            env_stats = {}
            if hasattr(eval_env, 'obs_count'):
                env_stats['entropy'] = self.marginal_entropy(eval_env.obs_count, eval_env.step_count)
                env_stats['tv_uniform'] = self.tv_uniform(eval_env.obs_count, eval_env.step_count)
                env_stats['kl_uniform'] = self.kl_uniform(eval_env.obs_count, eval_env.step_count)
            if hasattr(eval_env, 'transition_count'):
                trans_entropy, state_entropies = self.transition_entropy(eval_env.transition_count, eval_env.step_count)
                env_stats['transition_entropy'] = trans_entropy
                state_entropies_arr = np.array(list(state_entropies.values()))
                env_stats['sum_state_entropy'] = state_entropies_arr.sum()
                env_stats['min_state_entropy'] = state_entropies_arr.min()
                env_stats['mean_state_entropy'] = state_entropies_arr.mean()
            if hasattr(eval_env, 'hitting_time') and eval_env.hitting_time != 0:
                env_stats['hitting_time'] = eval_env.hitting_time
            if env_stats:
                logger.save_stats_csv(epoch, env_stats, fname='env_stats')
        mixing_time_dict = {}
        #print('EPOCH %d NUM STEPS: %d' % (epoch, eval_env.step_count))
        if hasattr(eval_env, 'mixing_time_periods'):
            for period in eval_env.mixing_time_periods:
                if epoch and epoch % period == 0 and len(eval_env.obs_counts) > period:
                    mixing_time_dict['mixing_time (k=%d)' % period] = self.tv_distance(eval_env, eval_env.obs_counts[-(period+1)], eval_env.obs_counts[-1], period)
                else:
                    mixing_time_dict['mixing_time (k=%d)' % period] = ''
            if mixing_time_dict:
                logger.save_stats_csv(epoch, mixing_time_dict, fname='mixing_time', numbered=False)

        logger.save_itr_params(epoch, snapshot)
        gt.stamp('saving', unique=False)
        self._log_stats(epoch, incl_expl=incl_expl)

        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def tv_distance(self, env, first, second, period):
        first_keys = set(first.keys())
        first_denom = env.step_count - period
        second_denom = env.step_count
        tv_distance = 0
        for k, v in first.items():
            tv_distance += 0.5 * np.abs(v / first_denom - second.get(k, 0) / second_denom)
            for k, v in second.items():
                if k not in first_keys:
                    tv_distance += 0.5 * v / second_denom
        return tv_distance

    def get_validation_returns(self, snapshot):
        policy = snapshot['evaluation/policy']
        policy = PolicyWrappedWithExplorationStrategy(
            EpsilonGreedy(self.eval_env.action_space, 0.1),
            policy
        )
        validation_envs = pickle.load(open(self.validation_envs_pkl, 'rb'))
        returns = np.zeros(len(validation_envs['envs']))
        for env_idx, env in enumerate(validation_envs['envs']):
            path = rollout(env, policy, self.validation_rollout_length)
            returns[env_idx] = path['rewards'].sum()
        return {'returns': returns.mean()}

    def validate(self, snapshot):
        """
        Collect list of stats for each validation env as dict of following format:
            'pickup_wood': [0, 15, 20] means you picked up a wood object at timesteps 0, 15, and 20.
        """
        policy = snapshot['evaluation/policy']
        if hasattr(policy, 'policy'):
            # if it's reset free, strip out the underlying policy from the exploration strategy
            policy = policy.policy
        policy = PolicyWrappedWithExplorationStrategy(
            EpsilonGreedy(self.eval_env.action_space, 0.1),
            policy
        )

        validation_envs = pickle.load(open(self.validation_envs_pkl, 'rb'))
        stats = [{} for _ in range(len(validation_envs['envs']))]
        for env_idx, env in enumerate(validation_envs['envs']):
            path = rollout(env, policy, self.validation_rollout_length)
            for typ in env.object_to_idx.keys():
                if typ not in ['empty', 'wall', 'tree']:
                    key = 'pickup_%s' % typ
                    last_val = 0
                    pickup_idxs = []
                    for t, env_info in enumerate(path['env_infos']):
                        count = env_info[key] - last_val
                        pickup_idxs.extend([t for _ in range(count)])
                        last_val = env_info[key]
                    stats[env_idx][key] = pickup_idxs
            for typ in env.interactions.values():
                key = 'made_%s' % typ
                last_val = 0
                made_idxs = []
                for t, env_info in enumerate(path['env_infos']):
                    count = env_info[key] - last_val
                    made_idxs.extend([t for _ in range(count)])
                    last_val = env_info[key]
                stats[env_idx][key] = made_idxs
        return stats

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def _log_stats(self, epoch, incl_expl=True):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        if incl_expl:
            """
            Exploration
            """
            logger.record_dict(
                self.expl_data_collector.get_diagnostics(),
                prefix='exploration/'
            )
            expl_paths = self.expl_data_collector.get_epoch_paths()
            if hasattr(self.expl_env, 'get_diagnostics'):
                logger.record_dict(
                    self.expl_env.get_diagnostics(expl_paths),
                    prefix='exploration/',
                )
            logger.record_dict(
                eval_util.get_generic_path_information(expl_paths),
                prefix="exploration/",
            )

        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if hasattr(self.eval_env, 'get_diagnostics'):
            logger.record_dict(
                self.eval_env.get_diagnostics(eval_paths),
                prefix='evaluation/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(eval_paths),
            prefix="evaluation/",
        )

        """
        Misc
        """
        gt.stamp('logging', unique=False)
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass
