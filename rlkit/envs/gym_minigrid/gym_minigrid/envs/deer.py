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


class DeerEnv(FoodEnvBase):
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
            deer_move_prob=0.2,
            deer_kill_easy_prob=0,
            deer_kill_easy_decay=1e-5,
            obs_vision=False,
            food_rate_decay=0.0,
            init_resources=None,
            gen_resources=True,
            # adjust lifespans to hold the number of resources const, based on resource gen probs. used for env sweeps.
            fixed_expected_resources=False,
            resource_prob=None,
            resource_prob_decay=None,
            resource_prob_min=None,
            place_schedule=None,
            make_rtype='sparse',
            rtype='default',
            lifespans=None,
            default_lifespan=0,
            task=None,
            rnd=False,
            cbe=False,
            seed_val=1,
            fixed_reset=False,
            end_on_task_completion=True,
            can_die=False,
            include_health=False,
            replenish_empty_resources=None,
            replenish_low_resources=None,
            # nonzero for reset case, 0 for reset free
            time_horizon=0,
            reset_hitting=True,
            **kwargs
    ):
        assert task is not None, 'Must specify task of form "make berry", "navigate 3 5", "pickup axe", etc.'
        # assert resource_prob is not None, 'Must specify resource generation probabilities'

        self.init_resources = init_resources or {}
        self.lifespans = lifespans or {}
        self.default_lifespan = default_lifespan
        self.food_rate_decay = food_rate_decay
        self.interactions = {
            # the 2 ingredients must be in alphabetical order
            # ('metal', 'wood'): 'axe',
            ('axe', 'deer'): 'food',
        }
        self.ingredients = {v: k for k, v in self.interactions.items()}
        self.gen_resources = gen_resources
        self.resource_prob = resource_prob or {}
        self.resource_prob_decay = resource_prob_decay or {}
        self.resource_prob_min = resource_prob_min or {}

        self.lifelong = time_horizon == 0
        
        self.reset_hitting = reset_hitting
        self.hitting_time = 0
        # deer stuff
        self.deer = []
        self.deer_move_prob = deer_move_prob
        self.deer_kill_easy_prob = deer_kill_easy_prob
        self.deer_kill_easy_decay = deer_kill_easy_decay

        # tuple of (bump, schedule), giving place_radius at time t = (t + bump) // schedule
        self.place_schedule = place_schedule
        # store whether the place_schedule has reached full grid radius yet, at which point it'll stop calling each time
        self.full_grid = False

        self.include_health = include_health
        self.replenish_empty_resources = replenish_empty_resources or []
        self.replenish_low_resources = replenish_low_resources or {}
        self.time_horizon = time_horizon
        # adjust lifespans if needed:
        if fixed_expected_resources:
            for type, num in self.init_resources.items():
                if type not in self.lifespans:
                    if not self.resource_prob.get(type, 0):
                        self.lifespans[type] = self.default_lifespan
                    else:
                        self.lifespans[type] = int(num / self.resource_prob[type])

        self.seed_val = seed_val
        self.fixed_reset = fixed_reset
        self.object_to_idx = {
            'empty': 0,
            'wall': 1,
            'food': 2,
            'wood': 3,
            'metal': 4,
            'deer': 5,
            'axe': 6,
            'monster': 7
        }

        # TASK stuff
        self.task = task
        self.task = task.split()  # e.g. 'pickup axe', 'navigate 3 5', 'make berry', 'make_lifelong axe'
        self.make_sequence = self.get_make_sequence()
        self.onetime_reward_sequence = [False for _ in range(len(self.make_sequence))]
        self.make_rtype = make_rtype

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
        self.end_on_task_completion = not self.lifelong

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
        self.actions = DeerEnv.Actions

        # stores info about picked up items
        self.info_last = {'pickup_%s' % k: 0 for k in self.object_to_idx.keys()
                          if k not in ['empty', 'wall', 'tree']}
        self.info_last.update({'made_%s' % v: 0 for v in self.interactions.values()})

        # stores visited locations for heat map
        self.visit_count = np.zeros((grid_size, grid_size), dtype=np.uint32)

        super().__init__(
            grid_size=grid_size,
            health_cap=health_cap,
            food_rate=food_rate,
            obs_vision=obs_vision,
            can_die=can_die,
            **kwargs
        )

        shape = None
        # TODO some of these shapes are wrong. fix by running through each branch and getting empirical obs shape
        if self.only_partial_obs:
            shape = (273,)
        elif self.grid_size == 32:
            if self.obs_vision:
                shape = (58569,)
            else:
                if self.fully_observed:
                    shape = (2067,)
                else:
                    shape = (2163,)
        elif self.grid_size == 16:
            if not self.obs_vision:
                if self.fully_observed:
                    shape = (631,)
        elif self.grid_size == 7:
            if not self.obs_vision and self.fully_observed:
                shape = (60,)
        elif self.grid_size == 8:
            if not self.obs_vision:
                if self.fully_observed:
                    shape = (1091,)
        # remove health component if not used
        if not include_health:
            shape = (shape[0] - 1,)

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

    def place_radius(self):
        assert self.place_schedule is not None, \
            '`place_schedule` must be specified as (bump, period), giving radius(t) = (t + bump) // period'
        return (self.step_count + self.place_schedule[0]) // self.place_schedule[1]

    def place_items(self):
        if not self.gen_resources:
            return
        counts = self.count_all_types()
        placed = set()
        for type, thresh in self.replenish_low_resources.items():
            if counts.get(type, 0) < thresh:
                new_obj = TYPE_TO_CLASS_ABS[type](lifespan=self.lifespans.get(type, self.default_lifespan))
                if self.place_schedule and not self.full_grid:
                    diam = self.place_radius()
                    if diam >= 2 * self.grid_size:
                        self.full_grid = True
                    self.place_prob(new_obj,
                                    1,
                                    top=(np.clip(self.agent_pos - diam // 2, 0, self.grid_size - 1)),
                                    size=(diam, diam))
                else:
                    self.place_prob(new_obj, 1)
                if type == 'deer':
                    self.deer.append(new_obj)
                placed.add(type)
        for type, prob in self.resource_prob.items():
            place_prob = max(self.resource_prob_min.get(type, 0),
                             prob - self.resource_prob_decay.get(type, 0) * self.step_count)
            if type in placed:
                place_prob = 0
            elif type in self.init_resources and not counts.get(type, 0) and type in self.replenish_empty_resources:
                # replenish resource if gone and was initially provided, or if resource is below the specified threshold
                place_prob = 1
            elif counts.get(type, 0) > (self.grid_size - 2) ** 2 // max(8, len(self.resource_prob)):
                # don't add more if it's already taking up over 1/8 of the space (lower threshold if >8 diff obj types being generated)
                place_prob = 0
            if self.place_schedule and not self.full_grid:
                diam = self.place_radius()
                if diam >= 2 * self.grid_size:
                    self.full_grid = True
                new_obj = TYPE_TO_CLASS_ABS[type](lifespan=self.lifespans.get(type, self.default_lifespan))
                if self.place_prob(new_obj,
                                place_prob,
                                top=(np.clip(self.agent_pos - diam // 2, 0, self.grid_size-1)),
                                size=(diam, diam)) and type == 'deer':
                    self.deer.append(new_obj)
            else:
                new_obj = TYPE_TO_CLASS_ABS[type](lifespan=self.lifespans.get(type, self.default_lifespan))
                if self.place_prob(new_obj,
                                place_prob) and type == 'deer':
                    self.deer.append(new_obj)

    def extra_gen_grid(self):
        for type, count in self.init_resources.items():
            if self.task and self.task[0] == 'pickup' and type == self.task[1]:
                for _ in range(count):
                    new_obj = TYPE_TO_CLASS_ABS[type]()
                    self.place_obj(new_obj)
                    if new_obj.type == 'deer':
                        self.deer.append(new_obj)
            else:
                for _ in range(count):
                    new_obj = TYPE_TO_CLASS_ABS[type](lifespan=self.lifespans.get(type, self.default_lifespan))
                    self.place_obj(new_obj)
                    if new_obj.type == 'deer':
                        self.deer.append(new_obj)

    def extra_step(self, action, matched):
        # Let deer move
        for deer in self.deer:
            if self._rand_float(0, 1) < self.deer_move_prob:
                new_pos = np.clip(deer.cur_pos + random.choice(list(DIR_TO_VEC.values())), 1, self.grid_size - 2)
                if not self.grid.get(*new_pos):
                    self.grid.set(*new_pos, deer)
                    self.grid.set(*deer.cur_pos, None)
                    deer.cur_pos = new_pos
        if matched:
            return matched

        agent_cell = self.grid.get(*self.agent_pos)
        matched = True

        # Collect resources. Add to shelf.
        if action == self.actions.mine:
            if agent_cell and agent_cell.can_mine(self):
                mined = False
                # check if food or other resource, which we're storing separately
                if self.include_health and agent_cell.food_value() > 0:
                    self.add_health(agent_cell.food_value())
                    mined = True
                    self.just_eaten_type = agent_cell.type
                else:
                    mined = self.add_to_shelf(agent_cell)

                if mined:
                    self.info_last['pickup_%s' % agent_cell.type] = self.info_last['pickup_%s' % agent_cell.type] + 1
                    self.grid.set(*self.agent_pos, None)

        # Consume stored food.
        elif action == self.actions.eat:
            pass

        # actions to use each element of inventory
        elif action == self.actions.place:
            self.place_act()

        else:
            matched = False

        return matched

    def place_act(self):
        agent_cell = self.grid.get(*self.agent_pos)
        if self.carrying is None:
            deer_kill_easy_prob = self.deer_kill_easy_prob - self.step_count * self.deer_kill_easy_decay
            if deer_kill_easy_prob > 0:
                # TODO   adding this to test changing dynamics as alternative to changing reward to make 3-tier task easier
                # killing deer without axe should have some (low) probability of success
                if agent_cell and agent_cell.type == 'deer':
                    if self._rand_float(0, 1) < deer_kill_easy_prob:
                        self.deer.remove(agent_cell)
                        new_type = 'food'
                        self.last_placed_on = agent_cell
                        self.just_placed_on = agent_cell
                        # replace existing obj with new obj
                        new_obj = TYPE_TO_CLASS_ABS[new_type](lifespan=self.lifespans.get(new_type, self.default_lifespan))
                        self.grid.set(*self.agent_pos, new_obj)
                        new_obj.cur_pos = self.agent_pos
                        self.made_obj_type = new_obj.type
                        self.just_made_obj_type = new_obj.type
                        self.info_last['made_%s' % new_type] = self.info_last['made_%s' % new_type] + 1

            # there's nothing to place
            return
        elif agent_cell is None:
            # there's nothing to combine it with, so just place it on the grid
            self.grid.set(*self.agent_pos, self.carrying)
            self.carrying.cur_pos = self.agent_pos
        else:
            # let's try to combine the placed object with the existing object
            interact_tup = tuple(sorted([self.carrying.type, agent_cell.type]))
            new_type = self.interactions.get(interact_tup, None)
            if not new_type:
                # the objects cannot be combined, no-op
                return
            else:
                if agent_cell.type == 'deer':
                    self.deer.remove(agent_cell)
                self.last_placed_on = agent_cell
                self.just_placed_on = agent_cell
                # replace existing obj with new obj
                new_obj = TYPE_TO_CLASS_ABS[new_type](lifespan=self.lifespans.get(new_type, self.default_lifespan))
                self.grid.set(*self.agent_pos, new_obj)
                new_obj.cur_pos = self.agent_pos
                self.made_obj_type = new_obj.type
                self.just_made_obj_type = new_obj.type
                self.info_last['made_%s' % new_type] = self.info_last['made_%s' % new_type] + 1
        # remove placed object from inventory
        self.carrying = None

    def add_to_shelf(self, obj):
        """ Returns whether adding to shelf succeeded """
        if self.carrying is None:
            self.carrying = obj
            return True
        return False

    def gen_shelf_obs(self):
        """ Return one-hot encoding of carried object type. """
        shelf_obs = np.zeros((1, len(self.object_to_idx)), dtype=np.uint8)
        if self.carrying is not None:
            shelf_obs[0, self.object_to_idx[self.carrying.type]] = 1
        return shelf_obs

    def step(self, action):
        self.just_made_obj_type = None
        self.just_eaten_type = None
        self.just_placed_on = None
        obs, reward, done, info = super().step(action, incl_health=self.include_health)
        shelf_obs = self.gen_shelf_obs()

        """ Generate obs """
        obs_grid_string = obs.tostring()
        extra_obs = shelf_obs.flatten()
        # magic number repeating shelf 8 times to fill up more of the obs
        extra_obs = np.repeat(extra_obs, 8)
        num_objs = np.repeat(self.info_last['pickup_%s' % self.task[1]], 8)
        obs = np.concatenate((obs, extra_obs, num_objs))

        """ Generate reward """
        solved = self.solved_task()
        if 'make' in self.task[0]:
            reward = self.get_make_reward()
            if self.task[0] == 'make':
                info.update({'progress': (self.max_make_idx + 1) / len(self.make_sequence)})
        else:
            reward = int(solved)

        """ Generate info """
        info.update({'health': self.health})
        info.update(self.info_last)
        if solved:
            if self.end_on_task_completion:
                done = True
            info.update({'solved': True})
            if self.lifelong:
                # remove obj so can keep making
                self.carrying = None
        else:
            info.update({'solved': False})
        if self.time_horizon and self.step_count % self.time_horizon == 0:
            done = True

        """ Exploration bonuses """
        self.obs_count[obs_grid_string] = self.obs_count.get(obs_grid_string, 0) + 1
        if self.cbe:
            reward += 1 / np.sqrt(self.obs_count[obs_grid_string])
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

        # funny ordering because otherwise we'd get the transpose due to how the grid indices work
        self.visit_count[self.agent_pos[1], self.agent_pos[0]] += 1
        if self.hitting_time == 0 and reward > 0:
            self.hitting_time = self.step_count
        return obs, reward, done, info

    def reset(self, seed=None, return_seed=False):
        prev_step_count = self.step_count
        self.deer = []
        if self.fixed_reset:
            self.seed(self.seed_val)
        else:
            if seed is None:
                seed = self._rand_int(0, 100000)
            self.seed(seed)
        obs = super().reset(incl_health=self.include_health)
        extra_obs = np.repeat(self.gen_shelf_obs(), 8)
        num_objs = np.repeat(self.info_last['pickup_%s' % self.task[1]], 8)
        obs = np.concatenate((obs, extra_obs, num_objs))

        if self.reset_hitting:
            self.hitting_time = 0
        else:
            self.step_count = prev_step_count
        self.pantry = []
        self.made_obj_type = None
        self.last_placed_on = None
        self.max_make_idx = -1
        self.last_idx = -1
        self.obs_count = {}
        self.info_last = {'pickup_%s' % k: 0 for k in self.object_to_idx.keys()
                          if k not in ['empty', 'wall', 'tree']}
        self.info_last.update({'made_%s' % v: 0 for v in self.interactions.values()})
        return (obs, seed) if return_seed else obs

    def solved_task(self):
        if 'make' in self.task[0]:
            return self.carrying is not None and self.carrying.type == self.task[1]
        elif self.task[0] == 'navigate':
            pos = np.array(self.task[1:])
            return np.array_equal(pos, self.agent_pos)
        elif self.task[0] == 'pickup':
            return self.carrying is not None and (self.carrying.type == self.task[1])
        else:
            raise NotImplementedError

    def get_make_reward(self):
        reward = 0
        if self.make_rtype == 'sparse':
            reward = POS_RWD * int(self.solved_task())
            if reward and self.lifelong:
                self.carrying = None
                self.num_solves += 1
        elif self.make_rtype in ['waypoint', 'dense']:
            carry_idx = self.make_sequence.index(
                self.carrying.type) if self.carrying and self.carrying.type in self.make_sequence else -1
            just_place_idx = self.make_sequence.index(
                self.just_placed_on.type) if self.just_placed_on and self.just_placed_on.type in self.make_sequence else -1
            just_made_idx = self.make_sequence.index(
                self.just_made_obj_type) if self.just_made_obj_type in self.make_sequence else -1
            idx = max(carry_idx, just_place_idx)
            true_idx = max(idx, self.max_make_idx - 1)
            cur_idx = max(carry_idx, just_made_idx)
            # print('carry: %d, place: %d, made: %d, j_made: %d, idx: %d, true: %d, cur: %d'
            #       % (carry_idx, just_place_idx, just_made_idx, just_made_idx, idx, true_idx, cur_idx))
            if carry_idx == len(self.make_sequence) - 1:
                reward = POS_RWD
                self.max_make_idx = -1
                self.num_solves += 1
                self.last_idx = -1
            elif just_made_idx > self.max_make_idx:
                reward = MED_RWD
                self.max_make_idx = just_made_idx
            elif idx == self.max_make_idx + 1:
                reward = MED_RWD
                self.max_make_idx = idx
            elif self.make_rtype == 'dense':
                if cur_idx < self.last_idx:
                    reward = NEG_RWD
                else:
                    next_pos = self.get_closest_obj_pos(self.make_sequence[true_idx + 1])
                    if next_pos is not None:
                        dist = np.linalg.norm(next_pos - self.agent_pos, ord=1)
                        reward = -0.01 * dist
                # else there is no obj of that type, so 0 reward
            if carry_idx != len(self.make_sequence) - 1:
                self.last_idx = cur_idx
        elif self.make_rtype in ['one-time', 'dense-fixed']:
            carry_idx = self.make_sequence.index(
                self.carrying.type) if self.carrying and self.carrying.type in self.make_sequence else -1
            just_place_idx = self.make_sequence.index(
                self.just_placed_on.type) if self.just_placed_on and self.just_placed_on.type in self.make_sequence else -1
            just_made_idx = self.make_sequence.index(
                self.just_made_obj_type) if self.just_made_obj_type in self.make_sequence else -1
            max_idx = max(carry_idx, just_place_idx)
            # print('carry: %d, j_place: %d, j_made: %d, max: %d, last: %d' % (carry_idx, just_place_idx, just_made_idx, max_idx, self.last_idx))
            if carry_idx == len(self.make_sequence) - 1:
                # exponent reasoning: 3rd obj in list should yield 100, 5th yields 10000, etc.
                reward = POS_RWD ** (carry_idx // 2)
                self.onetime_reward_sequence = [False for _ in range(len(self.make_sequence))]
                self.num_solves += 1
                # remove the created goal object
                self.carrying = None
                self.last_idx = -1
                if self.lifelong:
                    # otherwise messes with progress metric
                    self.max_make_idx = -1
            elif max_idx != -1 and not self.onetime_reward_sequence[max_idx]:
                # exponent reasoning: 3rd obj in list should yield 100, 5th yields 10000, etc.
                reward = POS_RWD ** (max_idx // 2)
                self.onetime_reward_sequence[max_idx] = True
            elif max(max_idx, just_made_idx) < self.last_idx:
                reward = -np.abs(NEG_RWD ** (self.last_idx // 2 + 1))
            elif self.make_rtype == 'dense-fixed':
                next_pos = self.get_closest_obj_pos(self.make_sequence[self.onetime_reward_sequence.index(False)])
                if next_pos is not None:
                    dist = np.linalg.norm(next_pos - self.agent_pos, ord=1)
                    reward = -0.01 * dist
            if max_idx > self.max_make_idx:
                self.max_make_idx = max_idx
            # only do this if it didn't just solve the task
            if carry_idx != len(self.make_sequence) - 1:
                self.last_idx = max_idx
        else:
            raise TypeError('Make reward type "%s" not recognized' % self.make_rtype)
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

    def dead_obj(self, i, j, obj):
        if obj.type == 'deer':
            self.deer.remove(obj)

    def decay_health(self):
        if self.include_health:
            super().decay_health()
