from rlkit.envs.gym_minigrid.gym_minigrid.minigrid_absolute import *
from rlkit.envs.gym_minigrid.gym_minigrid.register import register
from rlkit.envs.gym_minigrid.gym_minigrid.envs.getfood_base import FoodEnvBase
from rlkit.envs.gym_minigrid.gym_minigrid.minigrid_absolute import CELL_PIXELS, Food


class FoodEnvEasy(FoodEnvBase):
    """
    Pick up food to gain 1 health point,
    Lose 1 health point every `health_rate` timesteps,
    Get 1 reward per timestep
    """

    def __init__(self,
                 init_resources=None,
                 food_rate_decay=0.0,
                 lifespan=0,
                 her=False,
                 navigate=False,
                 **kwargs):
        self.init_resources = init_resources or {}
        self.food_rate_decay = food_rate_decay
        self.lifespan = lifespan
        self.her = her
        self.navigate = navigate

        super().__init__(**kwargs)

        if self.navigate:
            self.goal = self.place_obj(None)
            print(self.goal)

        if self.obs_vision:
            shape = (12481,)
        else:
            if self.fully_observed:
                shape = (131,)
            else:
                shape = (227,)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=shape,
            dtype='uint8'
        )

        if self.her or self.navigate:
            # position obs
            obs_space = spaces.Box(low=0, high=self.grid_size, shape=(2,), dtype='uint8')
            goal_space = spaces.Box(low=-self.grid_size, high=self.grid_size, shape=(2,), dtype='float16')
            if self.her:
                self.observation_space = spaces.Dict({
                    'observation': obs_space,
                    'achieved_goal': obs_space,
                    'desired_goal': obs_space
                })
            else:
                # navigate
                self.observation_space = obs_space
                self.goal_space = goal_space

    def extra_step(self, action, matched):
        self.food_rate += self.food_rate_decay

        if matched:
            return matched

        agent_cell = self.grid.get(*self.agent_pos)
        matched = True

        # Collect resources. In the case of this env, mining = instant health bonus.
        if action == self.actions.mine:
            if agent_cell and agent_cell.can_mine(self):
                self.grid.set(*self.agent_pos, None)
                self.add_health(agent_cell.food_value())
        else:
            matched = False

        return matched

    def extra_gen_grid(self):
        for type, count in self.init_resources.items():
            for _ in range(count):
                self.place_obj(TYPE_TO_CLASS_ABS[type]())

    def extra_reset(self):
        if self.her:
            self.goal_obs_her = np.random.randint(1, self.grid_size - 1, size=(2,))
            print(self.goal_obs_her)

    def place_items(self):
        if self.food_rate:
            self.place_prob(Food(lifespan=self.lifespan), 1 / self.food_rate)

    def step(self, action):
        obs, rwd, done, info = super().step(action)
        pos = np.array(self.agent_pos)
        if self.her:
            obs = {'observation': pos, 'desired_goal': self.goal_obs_her, 'achieved_goal': pos}
            rwd = self.compute_reward(self.agent_pos, self.goal_obs_her, info)
        elif self.navigate:
            obs = pos
            rwd = self.navigate_reward(obs)
            if rwd:
                self.goal = self.place_obj(None)
                print(self.goal)
        return obs, rwd, done, info

    def navigate_reward(self, obs):
        return int(np.array_equal(obs, self.goal))

    def compute_reward(self, achieved_goal, desired_goal, info):
        assert self.her, "`compute_reward` function should only be used for HER"

        return int(np.array_equal(achieved_goal, desired_goal))

    def reset(self):
        # this is done first so that agent_pos is updated
        super_reset = super().reset()
        pos = np.array(self.agent_pos)
        if self.her:
            return {'observation': pos, 'desired_goal': self.goal_obs_her, 'achieved_goal': pos}
        if self.navigate:
            return pos
        else:
            return super_reset

    def decay_health(self):
        if self.navigate:
            return
        super().decay_health()


class FoodEnvEasyCap50(FoodEnvEasy):
    pass


class FoodEnvEmptyFullObsHER(FoodEnvEasy):
    def __init__(self):
        super().__init__(fully_observed=True, her=True, food_rate=0)


class FoodEnvEmptyFullObsNavigate(FoodEnvEasy):
    def __init__(self):
        super().__init__(fully_observed=True, navigate=True, food_rate=0)


class FoodEnvEasyCap50Vision(FoodEnvEasy):
    def __init__(self):
        super().__init__(obs_vision=True)


class FoodEnvEasyCap100(FoodEnvEasy):
    def __init__(self):
        super().__init__(health_cap=100)


class FoodEnvEasyCap100Vision(FoodEnvEasy):
    def __init__(self):
        super().__init__(health_cap=100, obs_vision=True)


class FoodEnvEasyCap50Decay(FoodEnvEasy):
    def __init__(self):
        super().__init__(health_cap=50, food_rate_decay=0.005)


class FoodEnvEasyCap100Init10Decay(FoodEnvEasy):
    def __init__(self):
        super().__init__(health_cap=100, init_resources={'food': 10},
                         food_rate_decay=0.005)


class FoodEnvEasyCap100Init10DecayVision(FoodEnvEasy):
    def __init__(self):
        super().__init__(health_cap=100, init_resources={'food': 10},
                         food_rate_decay=0.005, obs_vision=True)


class FoodEnvEasyFood6Cap100Decay(FoodEnvEasy):
    def __init__(self):
        super().__init__(health_cap=100, food_rate=6, food_rate_decay=0.005)


class FoodEnvEasyFood6Cap2000Lifespan50FullObs(FoodEnvEasy):
    def __init__(self):
        super().__init__(food_rate=6, health_cap=2000, lifespan=50, fully_observed=True)


class FoodEnvEasyFood6Cap50DecayLifespan30(FoodEnvEasy):
    def __init__(self):
        super().__init__(food_rate=6, health_cap=50, food_rate_decay=0.005,
                         lifespan=30)


class FoodEnvEasyFood6Cap2000DecayLifespan30(FoodEnvEasy):
    def __init__(self):
        super().__init__(health_rate=10, food_rate=6, health_cap=2000, food_rate_decay=0.005,
                         lifespan=30)


class FoodEnvEasyFood6Cap2000DecayLifespan30FullObs(FoodEnvEasy):
    def __init__(self):
        super().__init__(food_rate=6, health_cap=2000, food_rate_decay=0.005,
                         lifespan=30, fully_observed=True)


register(
    id='MiniGrid-Food-8x8-Easy-Cap50-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasyCap50'
)

register(
    id='MiniGrid-Food-8x8-Empty-FullObs-HER-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEmptyFullObsHER'
)

register(
    id='MiniGrid-Food-8x8-Empty-FullObs-Navigate-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEmptyFullObsNavigate'
)


register(
    id='MiniGrid-Food-8x8-Easy-Cap50-Vision-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasyCap50Vision'
)

register(
    id='MiniGrid-Food-8x8-Easy-Cap100-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasyCap100'
)

register(
    id='MiniGrid-Food-8x8-Easy-Cap100-Vision-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasyCap100Vision'
)

register(
    id='MiniGrid-Food-8x8-Easy-Cap50-Decay-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasyCap50Decay'
)

register(
    id='MiniGrid-Food-8x8-Easy-Cap100-Init10-Decay-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasyCap100Init10Decay'
)

register(
    id='MiniGrid-Food-8x8-Easy-Cap100-Init10-Decay-Vision-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasyCap100Init10DecayVision'
)

register(
    id='MiniGrid-Food-8x8-Easy-Food6-Cap100-Decay-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasyFood6Cap100Decay'
)

register(
    id='MiniGrid-Food-8x8-Easy-Food6-Cap2000-Lifespan50-FullObs-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasyFood6Cap2000Lifespan50FullObs'
)

register(
    id='MiniGrid-Food-8x8-Easy-Food6-Cap50-Decay-Lifespan30-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasyFood6Cap50DecayLifespan30'
)

register(
    id='MiniGrid-Food-8x8-Easy-Food6-Cap2000-Decay-Lifespan30-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasyFood6Cap2000DecayLifespan30'
)

register(
    id='MiniGrid-Food-8x8-Easy-Food6-Cap2000-Decay-Lifespan30-FullObs-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasyFood6Cap2000DecayLifespan30FullObs'
)
