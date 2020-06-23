from collections import deque, OrderedDict

#from rlkit.envs.vae_wrapper import VAEWrappedEnv
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.rollout_functions import rollout, multitask_rollout, hierarchical_rollout, rollout_config
from rlkit.samplers.data_collector.base import PathCollector


class MdpPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                render=self._render and len(paths) == 0
            )
            path_len = len(path['actions'])
            #  : we don't want to skip incomplete paths, and in fact don't have a meaningful max path length
            # if (
            #         path_len != max_path_length
            #         and not path['terminals'][-1]
            #         and discard_incomplete_paths
            # ):
            #     break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
        )


class MdpPathCollectorConfig(MdpPathCollector):
    """ For resets to be able to retain env config by collecting seeds. """
    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths
    ):
        paths = []
        num_steps_collected = 0
        seed = None
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path, seed = rollout_config(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                render=self._render,# and len(paths) == 0,
                seed=seed
            )
            path_len = len(path['actions'])
            num_steps_collected += path_len
            paths.append(path)
            if path['env_infos'][-1]['solved']:
                # path terminated due to solve, so we can change env config
                seed = None
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths


class LifetimeMdpPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self.curr_env = env
        self.last_obs = None
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            continuing=False
    ):
        if not continuing:
            # reset held state re: env and obs since we're resetting now
            self.curr_env = self._env
            self.last_obs = None
        path, self.curr_env, self.last_obs = rollout(
            self.curr_env,
            self._policy,
            #  : this is not a typo
            max_path_length=num_steps,
            render=self._render,
            return_env_obs=True,
            continuing=continuing,
            obs=self.last_obs
        )
        path_len = len(path['actions'])
        self._num_paths_total += 1
        self._num_steps_total += path_len
        self._epoch_paths.append(path)
        return path

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
        )


class GoalConditionedPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            observation_key='observation',
            desired_goal_key='desired_goal',
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._render = render
        self._render_kwargs = render_kwargs
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            render=None
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = multitask_rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                render=self._render and len(paths) == 0,
                render_kwargs=self._render_kwargs,
                observation_key=self._observation_key,
                desired_goal_key=self._desired_goal_key,
                return_dict_obs=True,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )


class HierarchicalPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            setter,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._setter = setter
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._render = render
        self._render_kwargs = render_kwargs
        self._low_epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._high_epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

        self._low_num_steps_total = 0
        self._low_num_paths_total = 0
        self._high_num_steps_total = 0
        self._high_num_paths_total = 0

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths
    ):
        low_paths = []
        high_paths = []
        low_num_steps_collected = 0
        high_num_steps_collected = 0
        while low_num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - low_num_steps_collected,
            )
            low_path, high_path = hierarchical_rollout(
                self._env,
                self._policy,
                self._setter,
                max_path_length=max_path_length_this_loop,
                render=self._render and len(low_paths) == 0,
                render_kwargs=self._render_kwargs,
                return_dict_obs=True
            )

            low_path_len = len(low_path['actions'])
            high_path_len = len(high_path['actions'])
            if (
                    low_path_len != max_path_length
                    and not low_path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            low_num_steps_collected += low_path_len
            low_paths.append(low_path)
            high_num_steps_collected += high_path_len
            high_paths.append(high_path)

        self._low_num_paths_total += len(low_paths)
        self._low_num_steps_total += low_num_steps_collected
        self._low_epoch_paths.extend(low_paths)
        self._high_num_paths_total += len(high_paths)
        self._high_num_steps_total += high_num_steps_collected
        self._high_epoch_paths.extend(high_paths)
        return low_paths, high_paths

    def get_epoch_paths(self):
        return self._low_epoch_paths, self._high_epoch_paths

    def end_epoch(self, epoch):
        self._low_epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._high_epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        low_path_lens = [len(path['actions']) for path in self._low_epoch_paths]
        high_path_lens = [len(path['actions']) for path in self._high_epoch_paths]
        stats = OrderedDict([
            ('num low steps total', self._low_num_steps_total),
            ('num low paths total', self._low_num_paths_total),
            ('num high steps total', self._high_num_steps_total),
            ('num high paths total', self._high_num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "low path length",
            low_path_lens,
            always_show_all_stats=True,
        ))
        stats.update(create_stats_ordered_dict(
            "high path length",
            high_path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
            setter=self._setter
        )


class LifetimeHierarchicalPathCollector(HierarchicalPathCollector):
    def __init__(
            self,
            env,
            policy,
            setter,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
    ):
        super().__init__(env, policy, setter, max_num_epoch_paths_saved, render, render_kwargs)
        self.last_obs = None
        self.curr_env = None

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            continuing=False
    ):
        low_paths = []
        high_paths = []
        low_num_steps_collected = 0
        high_num_steps_collected = 0
        while low_num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - low_num_steps_collected,
            )
            low_path, high_path, self.curr_env, self.last_obs = hierarchical_rollout(
                self._env,
                self._policy,
                self._setter,
                max_path_length=max_path_length_this_loop,
                render=self._render and len(low_paths) == 0,
                render_kwargs=self._render_kwargs,
                return_dict_obs=True,
                return_env_obs=True,
                continuing=continuing,
                obs=self.last_obs
            )
            low_path_len = len(low_path['actions'])
            high_path_len = len(high_path['actions'])
            if (
                    low_path_len != max_path_length
                    and not low_path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            low_num_steps_collected += low_path_len
            low_paths.append(low_path)
        self._low_num_paths_total += len(low_paths)
        self._low_num_steps_total += low_num_steps_collected
        self._low_epoch_paths.extend(low_paths)
        self._high_num_paths_total += len(high_paths)
        self._high_num_steps_total += high_num_steps_collected
        self._high_epoch_paths.extend(high_paths)
        return low_paths, high_paths


