import cv2
from enum import IntEnum
from rlkit.envs.gym_minigrid.gym_minigrid.register import register
from gym import spaces
import numpy as np
from collections import defaultdict

from rlkit.envs.gym_minigrid.gym_minigrid.minigrid_absolute import MiniGridAbsoluteEnv, Food, GridAbsolute, CELL_PIXELS


class FoodEnvBase(MiniGridAbsoluteEnv):
    class Actions(IntEnum):
        # Absolute directions
        west = 0
        east = 1
        north = 2
        south = 3
        mine = 4

    def __init__(self,
                 agent_start_pos=(1, 1),
                 health_cap=50,
                 food_rate=4,
                 grid_size=8,
                 obs_vision=False,
                 reward_type='delta',
                 fully_observed=False,
                 only_partial_obs=False,
                 can_die=True,
                 one_hot_obs=True,
                 mixing_time_periods=[],
                 mixing_time_period_length=120,
                 **kwargs
                 ):
        self.agent_start_pos = agent_start_pos
        # self.agent_start_dir = agent_start_dir
        self.food_rate = food_rate
        self.health_cap = health_cap
        self.health = health_cap
        self.last_health = self.health
        self.obs_vision = obs_vision
        self.reward_type = reward_type
        self.fully_observed = fully_observed
        self.only_partial_obs = only_partial_obs
        self.can_die = can_die
        self.one_hot_obs = one_hot_obs
        # for conditional entropy of s' | s
        self.transition_count = {}
        self.prev_obs_string = ''
        # for mixing time
        self.mixing_time_periods = mixing_time_periods
        self.max_mixing_time_period = max(mixing_time_periods) if mixing_time_periods else 0
        self.mixing_time_period_length = mixing_time_period_length
        self.obs_counts = []

        if not hasattr(self, 'actions'):
            self.actions = FoodEnvBase.Actions
        super().__init__(
            # Set this to True for maximum speed
            see_through_walls=True,
            grid_size=grid_size,
            **kwargs
        )

    def _reward(self):
        if self.reward_type == 'survival':
            rwd = 1
        elif self.reward_type == 'delta':
            rwd = self.health - self.last_health
        elif self.reward_type == 'health':
            rwd = self.health
        else:
            assert False, "Reward type not matched"
        self.last_health = self.health
        return rwd

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = GridAbsolute(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        # self.grid.set(width - 2, height - 2, Goal())

        self.extra_gen_grid()

        # Place the agent
        if self.agent_start_pos is not None:
            self.start_pos = self.agent_start_pos
        # self.start_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = None

    def step(self, action, incl_health=True):
        done = False
        matched = super().step(action, override=True)
        # subclass-defined extra actions. if not caught by that, then unknown action
        if not self.extra_step(action, matched):
            assert False, "unknown action %d" % action

        # decrease health bar
        self.decay_health()
        # generate new food
        self.place_items()
        # generate obs after action is caught and food is placed. generate reward before death check
        img = self.get_img(onehot=self.one_hot_obs)
        full_img = self.get_full_img(scale=1 if self.fully_observed else 1 / 8, onehot=self.one_hot_obs)

        # NOTE: below not nec due to onehot being passed into two func calls above. but leaving here for now in case.
        # if self.one_hot_obs:
        # 	# ignore second channel since redundant (due to one-to-one mapping btwn color and obj type for now)
        # 	img = np.concatenate([np.eye(len(self.object_to_idx))[ch].transpose(2, 0, 1) for ch in img[:1]])
        # 	full_img = np.concatenate([np.eye(len(self.object_to_idx))[ch].transpose(2, 0, 1) for ch in full_img[:1]])

        rwd = self._reward()

        # tick on each grid item
        to_remove = []
        for j in range(0, self.grid.height):
            for i in range(0, self.grid.width):
                cell = self.grid.get(i, j)
                if cell is not None:
                    if not cell.step():
                        self.dead_obj(i, j, cell)
                        to_remove.append((i, j))
        for idxs in to_remove:
            self.grid.set(*idxs, None)

        # dead.
        if self.dead():
            done = True
        if self.fully_observed:
            if incl_health:
                obs = np.concatenate((full_img.flatten(), np.array([self.health]), np.array(self.agent_pos)))
            else:
                obs = np.concatenate((full_img.flatten(), np.array(self.agent_pos)))
        elif self.only_partial_obs:
            if incl_health:
                obs = np.concatenate((img.flatten(), np.array([self.health])))
            else:
                obs = img.flatten()
        else:
            if incl_health:
                obs = np.concatenate((img.flatten(), full_img.flatten(), np.array([self.health])))
            else:
                obs = np.concatenate((img.flatten(), full_img.flatten()))
        obs_string = obs.tostring()
        # transition count stuff
        self.transition_count.setdefault(hash(self.prev_obs_string), {})
        self.transition_count[hash(self.prev_obs_string)][hash(obs_string)] = 1 + self.transition_count[hash(self.prev_obs_string)].get(hash(obs_string), 0)
        # mixing time stuff
        if self.step_count % self.mixing_time_period_length == 0:
            self.obs_counts.append(self.obs_count.copy())
            if hasattr(self, 'obs_count') and self.mixing_time_periods and len(self.obs_counts) > self.max_mixing_time_period:
                self.obs_counts = self.obs_counts[-(self.max_mixing_time_period+1):]

        self.prev_obs_string = obs_string
        return obs, rwd, done, {}

    def reset(self, incl_health=True):
        super().reset()
        self.health = self.health_cap
        self.extra_reset()
        img = self.get_img(onehot=self.one_hot_obs)
        full_img = self.get_full_img(onehot=self.one_hot_obs)
        self.transition_count = {}

        # if self.one_hot_obs:
        # 	# ignore second channel since redundant (due to one-to-one mapping btwn color and obj type for now)
        # 	img = np.concatenate([np.eye(len(self.object_to_idx))[ch].transpose(2, 0, 1) for ch in img[:1]])
        # 	full_img = np.concatenate([np.eye(len(self.object_to_idx))[ch].transpose(2, 0, 1) for ch in full_img[:1]])
        if self.fully_observed:
            if incl_health:
                obs = np.concatenate((full_img.flatten(), np.array([self.health]), np.array(self.agent_pos)))
            else:
                obs = np.concatenate((full_img.flatten(), np.array(self.agent_pos)))
        elif self.only_partial_obs:
            if incl_health:
                obs = np.concatenate((img.flatten(), np.array([self.health])))
            else:
                obs = img.flatten()
        else:
            if incl_health:
                obs = np.concatenate((img.flatten(), full_img.flatten(), np.array([self.health])))
            else:
                obs = np.concatenate((img.flatten(), full_img.flatten()))
        self.prev_obs_string = obs.tostring()
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

    def extra_step(self, action, matched):
        return matched

    def extra_reset(self):
        pass

    def place_items(self):
        pass

    def extra_gen_grid(self):
        pass

    def place_prob(self, obj, prob, top=None, size=None):
        if np.random.binomial(1, prob):
            pos = self.place_obj(obj, top, size)
            obj.cur_pos = pos
            return True
        return False

    def decay_health(self):
        self.add_health(-1)

    def add_health(self, num):
        # clip health between 0 and cap after adjustment
        self.health = max(0, min(self.health_cap, self.health + num))

    def count_type(self, type):
        count = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = self.grid.get(i, j)
                if type is None and cell is None or cell is not None and cell.type == type:
                    count += 1
        return count

    def count_all_types(self):
        counts = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = self.grid.get(i, j)
                type = cell.type if cell is not None else ''
                counts[type] = counts.get(type, 0) + 1
        if hasattr(self, 'monsters'):
            counts['monster'] = len(self.monsters)
        return counts

    def exists_type(self, type):
        """ Check if object of type TYPE exists in current grid. """
        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                obj = self.grid.get(i, j)
                if obj and obj.type == type:
                    return True
        return False

    def dead(self):
        return self.can_die and self.health <= 0

    def dead_obj(self, i, j, obj):
        """ Called when OBJ dies at position (i, j). """
        pass

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['grid_render']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)


class FoodEnvEmptyFullObs(FoodEnvBase):
    def __init__(self):
        super().__init__(fully_observed=True)

    def decay_health(self):
        pass


register(
    id='MiniGrid-Food-8x8-Empty-FullObs-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEmptyFullObs'
)
