import cv2
from enum import IntEnum
from rlkit.envs.gym_minigrid.gym_minigrid.register import register
from gym import spaces
import numpy as np

from rlkit.envs.gym_minigrid.gym_minigrid.minigrid_absolute import MiniGridAbsoluteEnv, GridAbsolute, CELL_PIXELS


class UMaze(MiniGridAbsoluteEnv):
    class Actions(IntEnum):
        # Absolute directions
        west = 0
        east = 1
        north = 2
        south = 3

    def __init__(self,
                 agent_start_pos=(1, 1),
                 food_rate=4,
                 grid_size=10,
                 obs_vision=False,
                 reward_type='sparse',
                 fully_observed=False,
                 only_partial_obs=False,
                 one_hot_obs=True,
                 goal_pos=None,
                 **kwargs
                 ):
        self.agent_start_pos = agent_start_pos
        # self.agent_start_dir = agent_start_dir
        self.food_rate = food_rate
        self.obs_vision = obs_vision
        self.reward_type = reward_type
        self.fully_observed = fully_observed
        self.only_partial_obs = only_partial_obs
        self.one_hot_obs = one_hot_obs
        self.goal_pos = goal_pos or np.array([grid_size - 2, 1])

        # visitation count
        self.visit_count = np.zeros((grid_size, grid_size))

        self.object_to_idx = {
            'empty': 0,
            'wall': 1,
            # 'food': 2,
            # 'tree': 3,
            # 'metal': 4,
            # 'energy': 5,
            # 'axe': 6,
        }
        self.actions = UMaze.Actions

        super().__init__(
            # Set this to True for maximum speed
            see_through_walls=True,
            grid_size=grid_size,
            **kwargs
        )

        if self.fully_observed:
            # 2 for pos
            obs_dim = self.grid_size * self.grid_size * len(self.object_to_idx) + 2
        elif self.only_partial_obs:
            obs_dim = self.agent_view_size * self.agent_view_size * len(self.object_to_idx)
        else:
            obs_dim = self.grid_size * self.grid_size * len(self.object_to_idx) + 2 \
                      + self.agent_view_size * self.agent_view_size * len(self.object_to_idx)
        shape = (obs_dim,)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=shape,
            dtype='uint8'
        )

    def _reward(self):
        if self.reward_type == 'sparse':
            rwd = int(self.solved())
        elif self.reward_type == 'dense':
            rwd = -np.linalg.norm(self.goal_pos - self.agent_pos, ord=1)
        else:
            assert False, "Reward type not matched"
        return rwd

    def solved(self):
        return np.allclose(self.goal_pos, self.agent_pos)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = GridAbsolute(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # generate the middle wall
        self.grid.wall_rect(self.grid_size // 2 - 1, 0, 2, height - self.grid_size // 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.start_pos = self.agent_start_pos
        else:
            self.place_agent()

        self.mission = None

    def step(self, action, incl_health=True):
        super().step(action, override=True)
        img = self.get_img(onehot=self.one_hot_obs)
        full_img = self.get_full_img(scale=1 if self.fully_observed else 1 / 8, onehot=self.one_hot_obs)

        rwd = self._reward()
        done = self.solved()

        # update visitation count. transposed so that np array visually corresponds to grid
        self.visit_count[self.agent_pos[1], self.agent_pos[0]] += 1

        if self.fully_observed:
            obs = np.concatenate((full_img.flatten(), np.array(self.agent_pos)))
        elif self.only_partial_obs:
            obs = img.flatten()
        else:
            obs = np.concatenate((img.flatten(), full_img.flatten()))
        return obs, rwd, done, {}

    def reset(self, incl_health=True):
        super().reset()
        img = self.get_img(onehot=self.one_hot_obs)
        full_img = self.get_full_img(onehot=self.one_hot_obs)

        if self.fully_observed:
            obs = np.concatenate((full_img.flatten(), np.array(self.agent_pos)))
        elif self.only_partial_obs:
            obs = img.flatten()
        else:
            obs = np.concatenate((img.flatten(), full_img.flatten()))
        return obs

    def get_full_img(self, scale=1 / 8, onehot=False):
        """ Return the whole grid view """
        if self.obs_vision:
            full_img = self.get_full_obs_render(scale=scale)
        else:
            full_img = self.grid.encode(self, onehot=onehot)
        # NOTE: in case need to scale here instead of in above func call: return cv2.resize(full_img, (0, 0), fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
        return full_img

    def get_img(self, onehot=False):
        """ Return the agent view """
        if self.obs_vision:
            img = self.gen_obs(onehot=False)
            img = self.get_obs_render(img, CELL_PIXELS // 4)
        else:
            img = self.gen_obs(onehot=onehot)
        return img

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['grid_render']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)


class UMazeSparsePartial10(UMaze):
    def __init__(self):
        super().__init__(only_partial_obs=True, reward_type='sparse')


class UMazeDensePartial10(UMaze):
    def __init__(self):
        super().__init__(only_partial_obs=True, reward_type='dense')


class UMazeSparseFull10(UMaze):
    def __init__(self):
        super().__init__(fully_observed=True, reward_type='sparse')


class UMazeDenseFull10(UMaze):
    def __init__(self):
        super().__init__(fully_observed=True, reward_type='dense')


register(
    id='MiniGrid-UMaze-10x10-Sparse-Partial-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:UMazeSparsePartial10'
)

register(
    id='MiniGrid-UMaze-10x10-Dense-Partial-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:UMazeDensePartial10'
)

register(
    id='MiniGrid-UMaze-10x10-Sparse-Full-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:UMazeSparseFull10'
)

register(
    id='MiniGrid-UMaze-10x10-Dense-Full-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:UMazeDenseFull10'
)
