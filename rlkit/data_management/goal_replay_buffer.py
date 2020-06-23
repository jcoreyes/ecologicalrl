from gym.spaces import Discrete
import numpy as np
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.envs.env_utils import get_dim


class GoalSetterReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            goal_period,  # time btwn goals (length of low-level traj)
            env_info_sizes=None,
            dtype='float32'
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.goal_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes,
            dtype=dtype
        )

        # assumes low level gets same obs as high level
        self._traj_obs = np.zeros((max_replay_buffer_size, goal_period, get_dim(self._ob_space)))
        if isinstance(env.action_space, Discrete):
            self._traj_acs = np.zeros((max_replay_buffer_size, goal_period))
        else:
            self._traj_acs = np.zeros((max_replay_buffer_size, goal_period, get_dim(env.action_space)))

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, traj_obs, traj_acs, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        self._traj_obs[self._top] = traj_obs
        self._traj_acs[self._top] = traj_acs
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )

    def add_path(self, path):
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                traj_obs,
                traj_acs,
                agent_info,
                env_info
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["traj_obs"],
            path["traj_acs"],
            path["agent_infos"],
            path["env_infos"],
        )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                traj_obs=traj_obs,
                traj_acs=traj_acs,
                agent_info=agent_info,
                env_info=env_info,
            )
        self.terminate_episode()

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            traj_obs=self._traj_obs[indices],
            traj_acs=self._traj_acs[indices],
            next_observations=self._next_obs[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch


class GoalConditionedReplayBuffer(EnvReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None,
            dtype='float32'
    ):
        super().__init__(max_replay_buffer_size, env, env_info_sizes, dtype)

        self._goal_space = env.goal_space
        self._goals = np.zeros((self._max_replay_buffer_size, get_dim(self._goal_space)), dtype=self.dtype)

    def add_sample(self, observation, goal, action, reward, next_observation,
                   terminal, **kwargs):
        self._goals[self._top] = goal
        super().add_sample(observation, action, reward, terminal, next_observation, **kwargs)

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            goals=self._goals[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

    def add_path(self, path):
        for i, (
                obs,
                goal,
                action,
                reward,
                next_obs,
                terminal,
                agent_info,
                env_info
        ) in enumerate(zip(
            path["observations"],
            path["goals"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        )):
            self.add_sample(
                observation=obs,
                goal=goal,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
        self.terminate_episode()
