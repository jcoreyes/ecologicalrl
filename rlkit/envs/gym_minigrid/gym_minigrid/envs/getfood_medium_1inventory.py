from enum import IntEnum

from rlkit.envs.gym_minigrid.gym_minigrid.minigrid_absolute import *
from rlkit.envs.gym_minigrid.gym_minigrid.register import register
from rlkit.envs.gym_minigrid.gym_minigrid.envs.getfood_base import FoodEnvBase
from rlkit.envs.gym_minigrid.gym_minigrid.minigrid_absolute import CELL_PIXELS, MiniGridAbsoluteEnv, DIR_TO_VEC, \
    GridAbsolute
from rlkit.torch.core import torch_ify
from rlkit.torch.networks import Mlp
from torch.optim import Adam
from torch.nn import MSELoss

NEG_RWD = -100
POS_RWD = 100
MED_RWD = 1


class FoodEnvMedium1Inv(FoodEnvBase):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    class Actions(IntEnum):
        # Absolute directions
        west = 0
        east = 1
        north = 2
        south = 3
        # collect (but don't consume) an item
        mine = 4
        # consume a stored food item to boost health (does nothing if no stored food)
        eat = 5
        # place objects down
        place = 6

    def __init__(
            self,
            grid_size=32,
            health_cap=100,
            food_rate=4,
            max_pantry_size=50,
            obs_vision=False,
            food_rate_decay=0.0,
            init_resources=None,
            gen_resources=True,
            resource_prob=None,
            resource_prob_decay=None,
            make_rtype='sparse',
            rtype='default',
            goals=None,
            goal_period=300,
            lifespan=0,
            task=None,
            rnd=False,
            cbe=False,
            woodfood=True,
            seed_val=1,
            fixed_reset=False,
            end_on_task_completion=True,
            **kwargs
    ):
        assert 'goal' not in rtype or goals is not None, 'If using goal, must set one using `goals`.'
        assert 'goal' not in rtype or not task, 'Can\'t set task if using goal.'

        self.init_resources = init_resources or {}
        self.food_rate_decay = food_rate_decay
        self.lifespan = lifespan
        self.interactions = {
            ('energy', 'metal'): 'axe',
            # edible wood, used for health points
            ('axe', 'tree'): 'woodfood' if woodfood else 'wood',
        }
        self.ingredients = {v: k for k, v in self.interactions.items()}
        self.gen_resources = gen_resources
        self.resource_prob = resource_prob
        self.resource_prob_decay = resource_prob_decay
        self.seed_val = seed_val
        self.fixed_reset = fixed_reset
        self.object_to_idx = {
            'empty': 0,
            'wall': 1,
            'food': 2,
            'tree': 3,
            'metal': 4,
            'energy': 5,
            'axe': 6,
        }
        if woodfood:
            self.object_to_idx.update({'woodfood': 7})
        else:
            self.object_to_idx.update({'wood': 7})

        # TASK stuff
        self.task = task
        if self.task is not None:
            self.task = task.split()  # e.g. 'pickup axe', 'navigate 3 5', 'make wood', 'make_lifelong axe'
            self.make_sequence = self.get_make_sequence()
            self.onetime_reward_sequence = [False for _ in range(len(self.make_sequence))]
            self.make_rtype = make_rtype

        # how many of each obj the goal is
        self.goals = goals
        # temp goal (replaced by reset)
        self.goal = np.zeros((1, len(self.object_to_idx)), dtype=np.uint8)
        # how long the agent has to solve the goal
        self.goal_period = goal_period
        # idx of curr goal. if rtype 'goal' idx is always 0 since only 1 goal. if rtype 'goal_lifelong' then many goals
        self.goal_idx = 0
        self.rtype = rtype
        if 'goal' in self.rtype:
            self.inventory = np.zeros((1, len(self.object_to_idx)), dtype=np.uint8)
            self.goal = self.inventory.copy()
            for type, count in self.goals[self.goal_idx].items():
                self.goal[0, self.object_to_idx[type]] = count

        self.max_make_idx = -1
        self.last_idx = -1
        # most recent obj type made
        self.made_obj_type = None
        # obj type made in the last step, if any
        self.just_made_obj_type = None
        # obj type most recently placed on, if any
        self.last_placed_on = None
        # obj type placed on in the last step, if any
        self.just_placed_on = None
        # used for task 'make_lifelong'
        self.num_solves = 0
        self.end_on_task_completion = end_on_task_completion
        if (self.task and 'lifelong' in self.task[0]):# or self.rtype == 'goal_lifelong':
            self.end_on_task_completion = False

        # Exploration!
        assert not (cbe and rnd), "can't have both CBE and RND"
        # CBE
        self.cbe = cbe
        # RND
        self.rnd = rnd
        self.obs_count = {}
        # below two variables are to keep running count of stdev for RND normalization
        self.sum_rnd = 0
        self.sum_square_rnd = 0
        self.rnd_loss = MSELoss()

        # food
        self.pantry = []
        self.max_pantry_size = max_pantry_size
        # stores info for the current timestep
        self.info_last = {}
        self.actions = FoodEnvMedium1Inv.Actions

        super().__init__(
            grid_size=grid_size,
            health_cap=health_cap,
            food_rate=food_rate,
            obs_vision=obs_vision,
            **kwargs
        )

        shape = None
        # TODO some of these shapes are wrong. fix by running through each branch and getting empirical obs shape
        if self.only_partial_obs:
            shape = (665,)
        if self.grid_size == 32:
            if self.obs_vision:
                shape = (58969,)
            else:
                if self.fully_observed:
                    shape = (2459,)
                else:
                    shape = (2555,)
        elif self.grid_size == 16:
            if not self.obs_vision:
                if self.fully_observed:
                    shape = (923,)
        elif self.grid_size == 7:
            if not self.obs_vision and self.fully_observed:
                shape = (60,)
        elif self.grid_size == 8:
            if not self.obs_vision:
                if self.fully_observed:
                    shape = (1491,)
        if self.task:
            # exclude pantry and health
            shape = (shape[0] - 401,)
        elif 'goal' in self.rtype:
            # exclude pantry and health, but add in goal
            shape = (shape[0] - 401 + 64,)

        # if shape is None:
        # 	raise TypeError("Env configuration not supported")

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=shape,
            dtype='uint8'
        )

        if self.rnd:
            self.rnd_network = Mlp([128, 128], 32, self.observation_space.low.size)
            self.rnd_target_network = Mlp([128, 128], 32, self.observation_space.low.size)
            self.rnd_optimizer = Adam(self.rnd_target_network.parameters(), lr=3e-4)

    def get_make_sequence(self):
        def add_ingredients(obj, seq):
            for ingredient in self.ingredients.get(obj, []):
                add_ingredients(ingredient, seq)
            seq.append(obj)

        make_sequence = []
        goal_obj = self.task[1]
        add_ingredients(goal_obj, make_sequence)
        return make_sequence

    def place_items(self):
        counts = self.count_all_types()
        if self.gen_resources:
            if self.resource_prob:
                for type, prob in self.resource_prob.items():
                    if type in self.init_resources and not counts[type]:
                        # replenish resource if gone and was initially provided
                        place_prob = 1
                    elif self.resource_prob_decay and type in self.resource_prob_decay:
                        place_prob = max(0, prob - self.resource_prob_decay[type] * self.step_count)
                    elif counts[type] > (self.grid_size - 2) ** 2 // max(8, len(self.resource_prob)):
                        # don't add more if it's already taking up over 1/8 of the space (lower threshold if >10 diff obj types being generated)
                        place_prob = 0
                    else:
                        place_prob = prob
                    self.place_prob(TYPE_TO_CLASS_ABS[type](lifespan=self.lifespan), place_prob)
            else:
                self.place_prob(Food(lifespan=self.lifespan),
                                1 / (self.food_rate + self.step_count * self.food_rate_decay))
                self.place_prob(Metal(lifespan=self.lifespan), 1 / (2 * self.food_rate))
                self.place_prob(Energy(lifespan=self.lifespan), 1 / (2 * self.food_rate))
                self.place_prob(Tree(lifespan=self.lifespan), 1 / (3 * self.food_rate))

    def extra_gen_grid(self):
        for type, count in self.init_resources.items():
            if self.task and self.task[0] == 'pickup' and type == self.task[1]:
                for _ in range(count):
                    self.place_obj(TYPE_TO_CLASS_ABS[type]())
            else:
                for _ in range(count):
                    self.place_obj(TYPE_TO_CLASS_ABS[type](lifespan=self.lifespan))

    def extra_step(self, action, matched):
        if matched:
            return matched

        agent_cell = self.grid.get(*self.agent_pos)
        matched = True

        # Collect resources. Add to shelf.
        if action == self.actions.mine:
            if agent_cell and agent_cell.can_mine(self):
                mined = False
                # check if food or other resource, which we're storing separately
                if agent_cell.food_value() > 0:
                    if len(self.pantry) < self.max_pantry_size:
                        self.pantry.append(agent_cell)
                        mined = True
                else:
                    mined = self.add_to_shelf(agent_cell)

                if mined:
                    self.info_last.update({agent_cell.type: 1})
                    self.grid.set(*self.agent_pos, None)

        # Consume stored food.
        elif action == self.actions.eat:
            self.pantry.sort(key=lambda item: item.food_value(), reverse=True)
            if self.pantry:
                eaten = self.pantry.pop(0)
                if self.carrying and self.carrying.type == Axe().type:
                    self.add_health(eaten.food_value() * 2)
                else:
                    self.add_health(eaten.food_value())

        # actions to use each element of inventory
        elif action == self.actions.place:
            self.place_act()

        else:
            matched = False

        return matched

    def place_act(self):
        agent_cell = self.grid.get(*self.agent_pos)
        if self.carrying is None:
            # there's nothing to place
            return
        elif agent_cell is None:
            # there's nothing to combine it with, so just place it on the grid
            self.grid.set(*self.agent_pos, self.carrying)
        else:
            # let's try to combine the placed object with the existing object
            interact_tup = tuple(sorted([self.carrying.type, agent_cell.type]))
            new_type = self.interactions.get(interact_tup, None)
            if not new_type:
                # the objects cannot be combined, no-op
                return
            else:
                self.last_placed_on = agent_cell
                self.just_placed_on = agent_cell
                # replace existing obj with new obj
                new_obj = TYPE_TO_CLASS_ABS[new_type]()
                self.grid.set(*self.agent_pos, new_obj)
                self.made_obj_type = new_obj.type
                self.just_made_obj_type = new_obj.type
        # remove placed object from inventory
        self.carrying = None

    def add_to_shelf(self, obj):
        """ Returns whether adding to shelf succeeded """
        if self.carrying is None:
            self.carrying = obj
            return True
        return False

    def gen_pantry_obs(self):
        pantry_obs = np.zeros((self.max_pantry_size, len(self.object_to_idx)), dtype=np.uint8)
        pantry_idxs = [self.object_to_idx[obj.type] for obj in self.pantry]
        pantry_obs[np.arange(len(pantry_idxs)), pantry_idxs] = 1
        return pantry_obs

    def gen_shelf_obs(self):
        """ Return one-hot encoding of carried object type. """
        shelf_obs = np.zeros((1, len(self.object_to_idx)), dtype=np.uint8)
        if self.carrying is not None:
            shelf_obs[0, self.object_to_idx[self.carrying.type]] = 1
        return shelf_obs

    def step(self, action):
        self.just_made_obj_type = None
        self.just_placed_on = None
        incl_health = self.task is None and 'goal' not in self.rtype
        obs, reward, done, info = super().step(action, incl_health=incl_health)
        pantry_obs = self.gen_pantry_obs()
        shelf_obs = self.gen_shelf_obs()

        """ Generate obs """
        if not self.task and ('goal' not in self.rtype):
            extra_obs = np.concatenate((pantry_obs.flatten(), shelf_obs.flatten()))
            extra_obs_count_string = np.concatenate((pantry_obs.sum(axis=0), shelf_obs.sum(axis=0))).tostring()
        else:
            extra_obs = shelf_obs.flatten()
            # magic number repeating shelf 8 times to fill up more of the obs
            extra_obs = np.repeat(extra_obs, 8)
            extra_obs_count_string = shelf_obs.sum(axis=0).tostring()
        obs = np.concatenate((obs, extra_obs))
        if 'goal' in self.rtype:
            obs = np.concatenate((obs, np.repeat(self.goal, 8)))
            # if carrying an object that is part of the goal
            if self.carrying and self.carrying.type in self.goals[self.goal_idx] and \
                    self.inventory[0, self.object_to_idx[self.carrying.type]] <= self.goals[self.goal_idx][self.carrying.type]:
                self.inventory[0, self.object_to_idx[self.carrying.type]] += 1
                self.carrying = None

        """ Generate reward """
        solved = self.solved_task()
        if self.task and 'make' in self.task[0]:
            reward = self.get_make_reward()
            if self.task[0] == 'make':
                info.update({'progress': (self.max_make_idx + 1) / len(self.make_sequence)})
        elif 'goal' in self.rtype:
            reward = self.get_goal_reward()
        else:
            reward = int(solved)
        # necessary because part of the solve condition is updated in `get_goal_reward` (bad practice :()
        solved = self.solved_task()

        """ Generate info """
        info.update({'num_solves': self.num_solves})
        if solved:
            if self.end_on_task_completion:
                done = True
            info.update({'solved': True})
        else:
            info.update({'solved': False})

        """ Exploration bonuses """
        if self.cbe:
            self.obs_count[extra_obs_count_string] = self.obs_count.get(extra_obs_count_string, 0) + 1
            reward += 1 / np.sqrt(self.obs_count[extra_obs_count_string])
        elif self.rnd:
            torch_obs = torch_ify(extra_obs)
            true_rnd = self.rnd_network(torch_obs)
            pred_rnd = self.rnd_target_network(torch_obs)
            loss = self.rnd_loss(true_rnd, pred_rnd)

            self.rnd_optimizer.zero_grad()
            loss.backward()
            self.rnd_optimizer.step()
            # RND exploration bonus
            self.sum_rnd += loss
            self.sum_square_rnd += loss ** 2
            stdev = (self.sum_square_rnd / self.step_count) - (self.sum_rnd / self.step_count) ** 2
            reward += loss / (stdev * self.health_cap)

        return obs, reward, done, info

    def reset(self):
        if self.fixed_reset:
            self.seed(self.seed_val)
        incl_health = self.task is None and 'goal' not in self.rtype
        obs = super().reset(incl_health=incl_health)
        if 'goal' in self.rtype:
            extra_obs = np.repeat(self.gen_shelf_obs(), 8)
            goal_obs = np.repeat(self.goal, 8)
            obs = np.concatenate((obs, extra_obs.flatten(), goal_obs.flatten()))
        elif not self.task:
            obs = np.concatenate((obs, self.gen_pantry_obs().flatten(), self.gen_shelf_obs().flatten()))
        else:
            extra_obs = np.repeat(self.gen_shelf_obs(), 8)
            obs = np.concatenate((obs, extra_obs.flatten()))
        self.pantry = []
        self.made_obj_type = None
        self.last_placed_on = None
        self.max_make_idx = -1
        self.last_idx = -1
        self.obs_count = {}
        self.inventory = self.gen_shelf_obs()
        if 'goal' in self.rtype:
            self.steps_since_goal = 0
            self.goal_idx = -1
            self.goal = self.get_next_goal()
        return obs

    def solved_task(self):
        if self.task:
            if self.task[0] == 'navigate':
                pos = np.array(self.task[1:])
                return np.array_equal(pos, self.agent_pos)
            elif self.task[0] == 'pickup':
                return self.carrying is not None and (self.carrying.type == self.task[1])
            elif 'make' in self.task[0]:
                return self.carrying is not None and self.carrying.type == self.task[1]
        elif self.rtype == 'goal':
            deficit = np.maximum(self.goal - self.inventory, 0)
            return (deficit == 0).all()
        elif self.rtype == 'goal_lifelong':
            return (self.goal == 0).all()
        return False

    def get_make_reward(self):
        reward = 0
        if self.make_rtype == 'sparse':
            reward = POS_RWD * int(self.solved_task())
            if reward and 'lifelong' in self.task[0]:
                self.carrying = None
                self.num_solves += 1
        elif self.make_rtype in ['dense', 'waypoint']:
            carry_idx = self.make_sequence.index(
                self.carrying.type) if self.carrying and self.carrying.type in self.make_sequence else -1
            place_idx = self.make_sequence.index(
                self.last_placed_on.type) if self.last_placed_on and self.last_placed_on.type in self.make_sequence else -1
            made_idx = self.make_sequence.index(
                self.made_obj_type) - 1 if self.made_obj_type in self.make_sequence else -1
            just_made_idx = self.make_sequence.index(
                self.just_made_obj_type) if self.just_made_obj_type in self.make_sequence else -1
            idx = max(carry_idx, place_idx)
            true_idx = max(idx, made_idx, self.max_make_idx - 1)
            cur_idx = max(carry_idx, just_made_idx)

            if idx == len(self.make_sequence) - 1:
                reward = POS_RWD
                self.max_make_idx = idx
                self.num_solves += 1
            elif idx == self.max_make_idx + 1:
                reward = MED_RWD
                self.max_make_idx = idx
            # return reward
            elif made_idx > self.max_make_idx:
                reward = MED_RWD
                self.max_make_idx = made_idx
            elif self.make_rtype == 'dense':
                if cur_idx < self.last_idx:
                    reward = NEG_RWD
                else:
                    next_pos = self.get_closest_obj_pos(self.make_sequence[true_idx + 1])
                    if next_pos is not None:
                        dist = np.linalg.norm(next_pos - self.agent_pos, ord=1)
                        reward = -0.01 * dist
                # else there is no obj of that type, so 0 reward
            self.last_idx = cur_idx
        elif self.make_rtype == 'one-time':
            carry_idx = self.make_sequence.index(
                self.carrying.type) if self.carrying and self.carrying.type in self.make_sequence else -1
            just_place_idx = self.make_sequence.index(
                self.just_placed_on.type) if self.just_placed_on and self.just_placed_on.type in self.make_sequence else -1
            just_made_idx = self.make_sequence.index(
                self.just_made_obj_type) if self.just_made_obj_type in self.make_sequence else -1
            max_idx = max(carry_idx, just_place_idx)
            if carry_idx == len(self.make_sequence) - 1:
                # exponent reasoning: 3rd obj in list should yield 100, 5th yields 10000, etc.
                reward = POS_RWD ** (carry_idx // 2)
                self.onetime_reward_sequence = [False for _ in range(len(self.make_sequence))]
                self.num_solves += 1
                # remove the created goal object
                self.carrying = None
                self.last_idx = -1
                if self.task[0] == 'make_lifelong':
                    # otherwise messes with progress metric
                    self.max_make_idx = -1
            elif max_idx != -1 and not self.onetime_reward_sequence[max_idx]:
                # exponent reasoning: 3rd obj in list should yield 100, 5th yields 10000, etc.
                reward = POS_RWD ** (max_idx // 2)
                self.onetime_reward_sequence[max_idx] = True
            elif max_idx > self.max_make_idx:
                self.max_make_idx = max_idx
            elif max_idx < self.last_idx:
                reward = -np.abs(NEG_RWD ** (self.last_idx // 2 + 1))
            # only do this if it didn't just solve the task
            if carry_idx != len(self.make_sequence) - 1:
                self.last_idx = max_idx
        else:
            raise TypeError('Make reward type "%s" not recognized' % self.make_rtype)
        return reward

    def get_goal_reward(self):
        deficit = np.maximum(self.goal - self.inventory, 0)
        reward = -np.linalg.norm(deficit, ord=1)
        if self.rtype == 'goal':
            if reward == 0:
                reward += POS_RWD
        elif self.rtype == 'goal_lifelong':
            if reward == 0 or self.steps_since_goal and self.steps_since_goal % self.goal_period == 0:
                if reward == 0:
                    self.num_solves += 1
                    reward = POS_RWD
                self.steps_since_goal = 0
                self.inventory = np.zeros((1, len(self.object_to_idx)), dtype=np.uint8)
                self.goal = self.get_next_goal()
                if (self.goal == 0).all():
                    # done with all goals
                    reward = POS_RWD ** 2
        return reward

    def get_closest_obj_pos(self, type=None):
        def test_point(point, type):
            try:
                obj = self.grid.get(*point)
                if obj and (type is None or obj.type == type):
                    return True
            except AssertionError:
                # OOB grid access
                return False

        corner = np.array([0, -1])
        # range of max L1 distance on a grid of length self.grid_size - 2 (2 because of the borders)
        for i in range(0, 2 * self.grid_size - 5):
            # width of the fixed distance level set (diamond shape centered at agent pos)
            width = i + 1
            test_pos = self.agent_pos + corner * i
            for j in range(width):
                if test_point(test_pos, type):
                    return test_pos
                test_pos += np.array([1, 1])
            test_pos -= np.array([1, 1])
            for j in range(width):
                if test_point(test_pos, type):
                    return test_pos
                test_pos += np.array([-1, 1])
            test_pos -= np.array([-1, 1])
            for j in range(width):
                if test_point(test_pos, type):
                    return test_pos
                test_pos += np.array([-1, -1])
            test_pos -= np.array([-1, -1])
            for j in range(width):
                if test_point(test_pos, type):
                    return test_pos
                test_pos += np.array([1, -1])
        return None

    def get_next_goal(self):
        goal = np.zeros_like(self.goal)
        self.goal_idx += 1
        if self.goal_idx >= len(self.goals):
            return goal
        for type, count in self.goals[self.goal_idx].items():
            goal[0, self.object_to_idx[type]] = count
        return goal


    def decay_health(self):
        if (self.task and self.task[0] == 'make_lifelong') or ('goal' in self.rtype):
            return
        super().decay_health()


class FoodEnvMedium1InvCap50(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(health_cap=50)


class FoodEnvMedium1InvCap100(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(health_cap=100)


class FoodEnvMedium1InvCap100Vision(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(health_cap=10, obs_vision=True)


class FoodEnvMedium1InvCap500InitDecay(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(health_cap=5, food_rate_decay=0.01,
                         init_resources={
                             'axe': 8,
                             'woodfood': 5,
                             'food': 15
                         })


class FoodEnvMedium1InvCap500InitDecayLifespan80(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(health_cap=500, food_rate_decay=0.01, lifespan=80,
                         init_resources={
                             'axe': 8,
                             'woodfood': 5,
                             'food': 15
                         })


class FoodEnvMedium1InvCap500InitDecayFullObs(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(health_cap=500, food_rate_decay=0.01, fully_observed=True,
                         init_resources={
                             'axe': 8,
                             'woodfood': 5,
                             'food': 15
                         })


class FoodEnvMedium1InvCap500InitDecayLifespan80FullObs(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(health_cap=500, food_rate_decay=0.01, lifespan=80, fully_observed=True,
                         init_resources={
                             'axe': 8,
                             'woodfood': 5,
                             'food': 15
                         })


class FoodEnvMedium1InvCap2500InitDecayLifespan200FullObs(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(health_cap=2500, food_rate_decay=0.01, lifespan=200, fully_observed=True,
                         init_resources={
                             'axe': 8,
                             'woodfood': 5,
                             'food': 15
                         })


class FoodEnvMedium1InvCap1000InitDecayFullObsLifespan200Task(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(health_cap=1000, food_rate_decay=0.01, lifespan=200, fully_observed=True, task='make axe',
                         make_rtype='dense',
                         init_resources={
                             'axe': 8,
                             # 'woodfood': 5,
                             'food': 15,
                             'metal': 30,
                             'energy': 30
                         })


class FoodEnvMedium1InvCap1000InitDecayFullObsLifespan200TaskCBE(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(health_cap=1000, food_rate_decay=0.01, lifespan=200, fully_observed=True, task='pickup axe',
                         cbe=True,
                         init_resources={
                             'axe': 8,
                             'woodfood': 5,
                             'food': 15,
                         })


class FoodEnvMedium1Inv1TierDenseReward7(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=7, health_cap=1000, gen_resources=False, fully_observed=True, task='make energy',
                         make_rtype='dense',
                         init_resources={
                             'energy': 2
                         })


class FoodEnvMedium1Inv1TierDenseReward16(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=16, health_cap=1000, gen_resources=False, fully_observed=True, task='make energy',
                         make_rtype='dense',
                         init_resources={
                             'energy': 4
                         })


class FoodEnvMedium1Inv2TierDenseReward(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(health_cap=1000, gen_resources=False, fully_observed=True, task='make axe',
                         make_rtype='dense',
                         init_resources={
                             # 'food': 6,
                             'metal': 12,
                             'energy': 12
                         })


class FoodEnvMedium1Inv2TierDenseReward8(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=8, health_cap=1000, gen_resources=False, fully_observed=True, task='make axe',
                         make_rtype='dense', fixed_reset=True,
                         init_resources={
                             'metal': 4,
                             'energy': 4
                         })


class FoodEnvMedium1Inv2TierDenseRewardPartial8(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=8, health_cap=1000, gen_resources=False, fully_observed=False, task='make axe',
                         make_rtype='dense', fixed_reset=True,
                         init_resources={
                             'metal': 4,
                             'energy': 4
                         })


class FoodEnvMedium1Inv2TierDenseReward16(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=16, health_cap=1000, gen_resources=False, fully_observed=True, task='make axe',
                         make_rtype='dense', fixed_reset=True,
                         init_resources={
                             # 'food': 6,
                             'metal': 12,
                             'energy': 12
                         })


class FoodEnvMedium1Inv2TierWaypointReward(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(health_cap=1000, lifespan=200, fully_observed=True, task='make axe',
                         make_rtype='waypoint',
                         init_resources={
                             'food': 15,
                             'metal': 30,
                             'energy': 30
                         })


class FoodEnvMedium1Inv2TierSparseRewardCBE8(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=8, health_cap=1000, gen_resources=False, fully_observed=True, task='make axe',
                         make_rtype='sparse', fixed_reset=True, cbe=True,
                         init_resources={
                             # 'food': 6,
                             'metal': 4,
                             'energy': 4
                         })


class FoodEnvMedium1Inv2TierDenseRewardPartialRandom8(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=8, health_cap=1000, gen_resources=False, fully_observed=False, task='make axe',
                         make_rtype='dense', fixed_reset=False, only_partial_obs=True,
                         init_resources={
                             # 'food': 6,
                             'metal': 4,
                             'energy': 4
                         })


class FoodEnvMedium1Inv2TierDenseRewardRandom8(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=8, health_cap=1000, gen_resources=False, fully_observed=True, task='make axe',
                         make_rtype='dense', fixed_reset=False,
                         init_resources={
                             # 'food': 6,
                             'metal': 4,
                             'energy': 4
                         })


class FoodEnvMedium1Inv2TierDenseRewardPartialFixed8(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=8, health_cap=1000, gen_resources=False, fully_observed=False, task='make axe',
                         make_rtype='dense', fixed_reset=True, only_partial_obs=True,
                         init_resources={
                             # 'food': 6,
                             'metal': 1,
                             'energy': 1
                         })


class FoodEnvMedium1Inv2TierDenseRewardPartialFixedNoEnd8(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=8, health_cap=1000, gen_resources=False, fully_observed=False, task='make axe',
                         make_rtype='dense', fixed_reset=True, only_partial_obs=True, end_on_task_completion=False,
                         init_resources={
                             # 'food': 6,
                             'metal': 4,
                             'energy': 4
                         })


class FoodEnvMedium1Inv2TierSparseRewardPartial8Lifespan200(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=8, health_cap=1000, gen_resources=True, fully_observed=False,
                         task='make_lifelong axe',
                         make_rtype='sparse', only_partial_obs=True, lifespan=200,
                         init_resources={
                             'metal': 4,
                             'energy': 4
                         },
                         resource_prob={
                             'metal': 0.04,
                             'energy': 0.04
                         })


class FoodEnvMedium1Inv2TierOneTimeRewardPartial8Lifespan100(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=8, health_cap=1000, gen_resources=True, fully_observed=False,
                         task='make_lifelong axe',
                         make_rtype='one-time', only_partial_obs=True, lifespan=100,
                         init_resources={
                             'metal': 4,
                             'energy': 4
                         },
                         resource_prob={
                             'metal': 0.04,
                             'energy': 0.04
                         })


class FoodEnvMedium1Inv2TierOneTimeRewardPartial12Lifespan300(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=12, health_cap=1000, gen_resources=True, fully_observed=False,
                         task='make_lifelong axe',
                         make_rtype='one-time', only_partial_obs=True, lifespan=300,
                         init_resources={
                             'metal': 4,
                             'energy': 4
                         },
                         resource_prob={
                             'metal': 0.04,
                             'energy': 0.04
                         })


class FoodEnvMedium1Inv2TierOneTimeRewardPartial16Lifespan500(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=16, health_cap=1000, gen_resources=True, fully_observed=False,
                         task='make_lifelong axe',
                         make_rtype='one-time', only_partial_obs=True, lifespan=500,
                         init_resources={
                             'metal': 4,
                             'energy': 4
                         },
                         resource_prob={
                             'metal': 0.04,
                             'energy': 0.04
                         })


class FoodEnvMedium1Inv2TierOneTimeRewardPartial8(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=8, health_cap=1000, gen_resources=False, fully_observed=False, task='make axe',
                         make_rtype='one-time', only_partial_obs=True,
                         init_resources={
                             'metal': 4,
                             'energy': 4
                         })


class FoodEnvMedium1Inv2TierOneTimeRewardPartial8Lifespan400(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=8, health_cap=1000, gen_resources=True, fully_observed=False,
                         task='make_lifelong axe',
                         make_rtype='one-time', only_partial_obs=True, lifespan=400,
                         init_resources={
                             'metal': 4,
                             'energy': 4
                         },
                         resource_prob={
                             'metal': 0.01,
                             'energy': 0.01
                         })


class FoodEnvMedium1Inv2TierOneTimeRewardPartial16Lifespan400(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=16, health_cap=1000, gen_resources=True, fully_observed=False,
                         task='make_lifelong axe',
                         make_rtype='one-time', only_partial_obs=True, lifespan=400,
                         init_resources={
                             'metal': 16,
                             'energy': 16
                         },
                         resource_prob={
                             'metal': 0.04,
                             'energy': 0.04
                         })


class FoodEnvMedium1Inv3TierSparseRewardPartial8Lifespan500(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=8, health_cap=1000, gen_resources=True, fully_observed=False,
                         task='make_lifelong wood', woodfood=False,
                         make_rtype='sparse', only_partial_obs=True, lifespan=500,
                         init_resources={
                             'metal': 4,
                             'energy': 4,
                             'tree': 2
                         },
                         resource_prob={
                             'metal': 0.04,
                             'energy': 0.04,
                             'tree': 0.02
                         })


class FoodEnvMedium1Inv3TierOneTimeRewardPartial8Lifespan500(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=8, health_cap=1000, gen_resources=True, fully_observed=False,
                         task='make_lifelong wood', woodfood=False,
                         make_rtype='one-time', only_partial_obs=True, lifespan=500,
                         init_resources={
                             'metal': 4,
                             'energy': 4,
                             'tree': 2
                         },
                         resource_prob={
                             'metal': 0.04,
                             'energy': 0.04,
                             'tree': 0.02
                         })


class FoodEnvMedium1Inv3TierSparseRewardPartial8IntermediateLifespan500(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=8, health_cap=1000, gen_resources=True, fully_observed=False,
                         task='make_lifelong wood', woodfood=False,
                         make_rtype='sparse', only_partial_obs=True, lifespan=500,
                         init_resources={
                             'metal': 4,
                             'energy': 4,
                             'tree': 2,
                             'axe': 2
                         },
                         resource_prob={
                             'metal': 0.04,
                             'energy': 0.04,
                             'tree': 0.02,
                             'axe': 0.02
                         },
                         resource_prob_decay={
                             'axe': 1e-6
                         })


class FoodEnvMedium1Inv3TierOneTimeRewardPartial8IntermediateLifespan500(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=8, health_cap=1000, gen_resources=True, fully_observed=False,
                         task='make_lifelong wood', woodfood=False,
                         make_rtype='one-time', only_partial_obs=True, lifespan=500,
                         init_resources={
                             'metal': 4,
                             'energy': 4,
                             'tree': 2,
                             'axe': 2
                         },
                         resource_prob={
                             'metal': 0.04,
                             'energy': 0.04,
                             'tree': 0.02,
                             'axe': 0.02
                         },
                         resource_prob_decay={
                             'axe': 1e-6
                         })


class FoodEnvMedium1Inv3TierDenseReward8(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=8, health_cap=1000, gen_resources=False, fully_observed=True, task='make wood',
                         make_rtype='dense', woodfood=False, fixed_reset=True,
                         init_resources={
                             'metal': 4,
                             'energy': 4,
                             'tree': 4
                         })


class FoodEnvMedium1Inv3TierDenseReward(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(health_cap=1000, lifespan=200, fully_observed=True, task='make wood', woodfood=False,
                         make_rtype='dense',
                         init_resources={
                             'food': 15,
                             'metal': 30,
                             'energy': 30,
                             'tree': 15
                         })


class FoodEnvMedium1Inv3TierWaypointReward(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(health_cap=1000, lifespan=200, fully_observed=True, task='make wood', woodfood=False,
                         make_rtype='waypoint',
                         init_resources={
                             'food': 15,
                             'metal': 30,
                             'energy': 30,
                             'tree': 15
                         })


class FoodEnvMedium1Inv1TierGoalFixed8(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=8, health_cap=1000, gen_resources=False, fully_observed=False, rtype='goal',
                         fixed_reset=True, only_partial_obs=True,
                         goals=[{
                             'energy': 3
                         }],
                         init_resources={
                             'energy': 6
                         })


class FoodEnvMedium1Inv2TierGoalFixed8(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=8, health_cap=1000, gen_resources=False, fully_observed=False, rtype='goal',
                         fixed_reset=True, only_partial_obs=True,
                         goals=[{
                             'axe': 2
                         }],
                         init_resources={
                             'energy': 4,
                             'metal': 4
                         })


class FoodEnvMedium1Inv2TierGoalRandom8(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=8, health_cap=1000, gen_resources=False, fully_observed=False, rtype='goal',
                         fixed_reset=False, only_partial_obs=True,
                         goals=[{
                             'axe': 2
                         }],
                         init_resources={
                             'energy': 4,
                             'metal': 4
                         })


class FoodEnvMedium1InvGoalRandom8(FoodEnvMedium1Inv):
    def __init__(self):
        super().__init__(grid_size=8, health_cap=1000, gen_resources=True, fully_observed=False, rtype='goal_lifelong',
                         fixed_reset=False, only_partial_obs=True, lifespan=100,
                         goals=[{
                             'energy': 2
                         }] * 5 + [{
                             'metal': 2
                         }] * 5 + [{
                             'axe': 1
                         }] * 10 + [{
                             'axe': 2
                         }] * 1000,
                         resource_prob={
                             'metal': 0.04,
                             'energy': 0.04,
                         },
                         init_resources={
                             'energy': 4,
                             'metal': 4
                         })


register(
    id='MiniGrid-Food-32x32-Medium-1Inv-Cap50-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1InvCap50'
)

register(
    id='MiniGrid-Food-32x32-Medium-1Inv-Cap100-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1InvCap100'
)

register(
    id='MiniGrid-Food-32x32-Medium-1Inv-Cap100-Vision-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1InvCap100Vision'
)

register(
    id='MiniGrid-Food-32x32-Medium-1Inv-Cap500-Init-Decay-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1InvCap500InitDecay'
)

register(
    id='MiniGrid-Food-32x32-Medium-1Inv-Cap500-Init-Decay-Lifespan80-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1InvCap500InitDecayLifespan80'
)

register(
    id='MiniGrid-Food-32x32-Medium-1Inv-Cap100-Init-Decay-FullObs-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1InvCap500InitDecayFullObs'
)

register(
    id='MiniGrid-Food-32x32-Medium-1Inv-5and4-Cap100-Init-Decay-FullObs-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv5and4Cap100InitDecayFullObs'
)

register(
    id='MiniGrid-Food-32x32-Medium-1Inv-Cap500-Init-Decay-Lifespan80-FullObs-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1InvCap500InitDecayLifespan80FullObs'
)

register(
    id='MiniGrid-Food-32x32-Medium-1Inv-Cap2500-Init-Decay-Lifespan200-FullObs-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1InvCap2500InitDecayLifespan200FullObs'
)

register(
    id='MiniGrid-Food-32x32-Medium-1Inv-Cap1000-Init-Decay-FullObs-Lifespan200-Task-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1InvCap1000InitDecayFullObsLifespan200Task'
)

register(
    id='MiniGrid-Food-32x32-Medium-1Inv-Cap1000-Init-Decay-FullObs-Lifespan200-Task-CBE-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1InvCap1000InitDecayFullObsLifespan200TaskCBE'
)

register(
    id='MiniGrid-Food-7x7-Medium-1Inv-1Tier-Dense-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv1TierDenseReward7'
)

register(
    id='MiniGrid-Food-16x16-Medium-1Inv-1Tier-Dense-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv1TierDenseReward16'
)

register(
    id='MiniGrid-Food-32x32-Medium-1Inv-2Tier-Dense-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierDenseReward'
)

register(
    id='MiniGrid-Food-8x8-Medium-1Inv-2Tier-Dense-Partial-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierDenseRewardPartial8'
)

register(
    id='MiniGrid-Food-8x8-Medium-1Inv-2Tier-Dense-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierDenseReward8'
)

register(
    id='MiniGrid-Food-16x16-Medium-1Inv-2Tier-Dense-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierDenseReward16'
)

register(
    id='MiniGrid-Food-32x32-Medium-1Inv-2Tier-Waypoint-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierWaypointReward'
)

register(
    id='MiniGrid-Food-8x8-Medium-1Inv-2Tier-Sparse-CBE-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierSparseRewardCBE8'
)

register(
    id='MiniGrid-Food-8x8-Medium-1Inv-2Tier-Dense-Partial-Random-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierDenseRewardPartialRandom8'
)

register(
    id='MiniGrid-Food-8x8-Medium-1Inv-2Tier-Dense-Partial-Random-1Resource-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv',
    kwargs=dict(
        grid_size=8,
        # start agent at random pos
        agent_start_pos=None,
        health_cap=1000,
        gen_resources=False,
        fully_observed=False,
        task='make axe',
        make_rtype='dense',
        fixed_reset=False,
        only_partial_obs=True,
        init_resources={
            'metal': 1,
            'energy': 1
        }
    )
)

register(
    id='MiniGrid-Food-8x8-Medium-1Inv-2Tier-Dense-Partial-Lifespan200-1Resource-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv',
    kwargs=dict(
        grid_size=8,
        health_cap=1000,
        fully_observed=False,
        task='make_lifelong axe',
        make_rtype='one-time',
        fixed_reset=False,
        only_partial_obs=True,
        init_resources={
            'metal': 1,
            'energy': 1
        },
        resource_prob={
            'metal': 0,
            'energy': 0
        },
        gen_resources=True,
        lifespan=200
    )
)

register(
    id='MiniGrid-Food-8x8-Medium-1Inv-2Tier-Dense-Random-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierDenseRewardRandom8'
)

register(
    id='MiniGrid-Food-8x8-Medium-1Inv-2Tier-Dense-Partial-Fixed-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierDenseRewardPartialFixed8'
)

register(
    id='MiniGrid-Food-8x8-Medium-1Inv-2Tier-Dense-Partial-Fixed-NoEnd-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierDenseRewardPartialFixedNoEnd8'
)

register(
    id='MiniGrid-Food-8x8-Medium-1Inv-2Tier-Sparse-Partial-Lifespan200-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierSparseRewardPartial8Lifespan200'
)

register(
    id='MiniGrid-Food-8x8-Medium-1Inv-2Tier-OneTime-Partial-Lifespan100-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierOneTimeRewardPartial8Lifespan100'
)

register(
    id='MiniGrid-Food-12x12-Medium-1Inv-2Tier-OneTime-Partial-Lifespan300-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierOneTimeRewardPartial12Lifespan300'
)

register(
    id='MiniGrid-Food-16x16-Medium-1Inv-2Tier-OneTime-Partial-Lifespan500-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierOneTimeRewardPartial16Lifespan500'
)

register(
    id='MiniGrid-Food-8x8-Medium-1Inv-2Tier-OneTime-Partial-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierOneTimeRewardPartial8'
)

register(
    id='MiniGrid-Food-8x8-Medium-1Inv-2Tier-OneTime-Partial-Lifespan400-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierOneTimeRewardPartial8Lifespan400'
)

register(
    id='MiniGrid-Food-16x16-Medium-1Inv-2Tier-OneTime-Partial-Lifespan400-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierOneTimeRewardPartial16Lifespan400'
)

register(
    id='MiniGrid-Food-8x8-Medium-1Inv-3Tier-Sparse-Partial-Lifespan500-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv3TierSparseRewardPartial8Lifespan500'
)

register(
    id='MiniGrid-Food-8x8-Medium-1Inv-3Tier-OneTime-Partial-Lifespan500-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv3TierOneTimeRewardPartial8Lifespan500'
)

register(
    id='MiniGrid-Food-8x8-Medium-1Inv-3Tier-Sparse-Partial-Intermediate-Lifespan500-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv3TierSparseRewardPartial8IntermediateLifespan500'
)

register(
    id='MiniGrid-Food-8x8-Medium-1Inv-3Tier-OneTime-Partial-Intermediate-Lifespan500-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv3TierOneTimeRewardPartial8IntermediateLifespan500'
)


register(
    id='MiniGrid-Food-8x8-Medium-1Inv-3Tier-Dense-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv3TierDenseReward8'
)

register(
    id='MiniGrid-Food-32x32-Medium-1Inv-3Tier-Dense-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv3TierDenseReward'
)

register(
    id='MiniGrid-Food-32x32-Medium-1Inv-3Tier-Waypoint-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv3TierWaypointReward'
)

register(
    id='MiniGrid-Food-8x8-Medium-1Inv-1Tier-Goal-Fixed-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv1TierGoalFixed8'
)

register(
    id='MiniGrid-Food-8x8-Medium-1Inv-2Tier-Goal-Fixed-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierGoalFixed8'
)

register(
    id='MiniGrid-Food-8x8-Medium-1Inv-2Tier-Goal-Random-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierGoalRandom8'
)

register(
    id='MiniGrid-Food-8x8-Medium-1Inv-GoalLifetime-Random-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1InvGoalRandom8'
)
