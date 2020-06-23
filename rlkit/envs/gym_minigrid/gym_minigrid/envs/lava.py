from gym_minigrid.envs.tools import ToolsEnv
from gym_minigrid.minigrid_absolute import Lava


class LavaEnv(ToolsEnv):
    def __init__(self, num_lava=3, lava_timeout=0, lava_timeout_increase=0.0, lava_penalty=0, **kwargs):
        self.num_lava = num_lava
        self.lava_timeout = lava_timeout
        self.lava_timeout_increase = lava_timeout_increase
        self.lava_penalty = lava_penalty
        self.lava_time_active = 0
        self.object_to_idx = {
            'empty': 0,
            'wall': 1,
            'food': 2,
            'wood': 3,
            'metal': 4,
            'tree': 5,
            'axe': 6,
            'berry': 7,
            'lava': 8
        }
        super().__init__(**kwargs)

    def extra_gen_grid(self):
        for _ in range(self.num_lava):
            self.place_obj(Lava())

    def step(self, action):
        penalty = 0
        # take care of lava behavior
        self.lava_timeout += self.lava_timeout_increase
        agent_cell = self.grid.get(*self.agent_pos)
        if self.lava_time_active >= self.lava_timeout:
            # the agent did its time, can now unfreeze
            self.can_move = True
            self.lava_time_active = 0
        elif agent_cell and isinstance(agent_cell, Lava):
            self.can_move = False
            self.lava_time_active += 1
        # normal step
        obs, reward, done, info = super().step(action)
        # check whether agent moved to lava for penalty
        agent_cell = self.grid.get(*self.agent_pos)
        if agent_cell and isinstance(agent_cell, Lava):
            penalty = self.lava_penalty
        reward -= penalty
        return obs, reward, done, info
