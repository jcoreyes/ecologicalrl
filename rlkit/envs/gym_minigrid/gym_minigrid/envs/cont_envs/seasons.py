from enum import IntEnum

from gym import spaces
from gym_minigrid.envs import FoodEnvBase
from rlkit.envs.gym_minigrid.gym_minigrid.register import register
from rlkit.envs.gym_minigrid.gym_minigrid.minigrid_absolute import *

NEG_RWD = -100
POS_RWD = 100
MED_RWD = 1


class Seasons(FoodEnvBase):
    """
    Seasons of different resources, with soft transitions
    (existing items not cleared out, but die out eventually)

    ** Env sweep hyperparams:
        ** resets:      boolean
        ** reward_type: 'dense' or 'waypoints' or 'health' (ordered by increasing sparsity)
        ** task_seq:    boolean
        ** lifespan:    100, 300, 500
        ** view_size:   3, 5, 7
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
            # seasons
            seasons,
            season_lens,
            season_seq,
            task_seq,

            grid_size=32,
            obs_vision=False,

            health_cap=1000,
            max_pantry_size=50,

            # Ass. 1: reset free
            reset_free=False,
            fixed_reset=False,  # whether to reseed with same random value

            # Ass. 2: reward
            rtype='health',

            # Ass. 4: stochasticity
            lifespan=0,
            init_resources=None,
            gen_resources=True,
            resource_prob=None,
            resource_prob_decay=None,

            # Ass. 5: partially observed view size
            agent_view_size=5,

            seed_val=1,
            **kwargs
    ):
        # Season Stuff
        # list of dictionaries corresponding to resource generation probabilities per season
        self.seasons = seasons
        # number of timesteps each season should last
        self.season_lens = season_lens
        # the build sequence for each season
        self.season_seq = season_seq
        # sequence of tasks per season, selected from "pickup" or "make", based on whether final obj should be picked up (e.g. wood) or made (e.g. tree)
        self.task_seq = task_seq
        # which season is it right now
        self.season_idx = 0
        # how many steps have we taken in the current season
        self.season_steps = 0

        self.reset_free = reset_free
        self.fixed_reset = fixed_reset
        self.rtype = rtype

        self.init_resources = init_resources or {}
        self.lifespan = lifespan
        self.interactions = {
            ('seed', 'water'): 'plant',
            ('plant', 'sun'): 'tree',
            ('energy', 'metal'): 'axe',
            ('axe', 'tree'): 'bigfood'
        }
        self.ingredients = {v: k for k, v in self.interactions.items()}
        self.gen_resources = gen_resources
        self.resource_prob = resource_prob
        self.resource_prob_decay = resource_prob_decay
        self.seed_val = seed_val
        self.object_to_idx = {
            'empty': 0,
            'wall': 1,
            'bigfood': 2,
            'tree': 3,
            'metal': 4,
            'energy': 5,
            'axe': 6,
            'water': 7,
            'seed': 8,
            'plant': 9,
            'sun': 10,
        }

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

        # food
        self.pantry = []
        self.max_pantry_size = max_pantry_size
        self.actions = Seasons.Actions

        # stores info about picked up items
        self.info_last = {'pickup_%s' % k: 0 for k in self.object_to_idx.keys()
                          if k in TYPE_TO_CLASS_ABS and TYPE_TO_CLASS_ABS[k]().can_mine(self)}
        self.info_last.update({'made_%s' % v: 0 for v in self.interactions.values()})

        super().__init__(
            grid_size=grid_size,
            health_cap=health_cap,
            obs_vision=obs_vision,
            agent_view_size=agent_view_size,
            **kwargs
        )

        shape = (1,)
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
                else:
                    shape = (500,)

        # if shape is None:
        # 	raise TypeError("Env configuration not supported")

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=shape,
            dtype='uint8'
        )

    def place_items(self):
        counts = self.count_all_types()
        if self.gen_resources:
            for type, prob in self.seasons[self.season_idx].items():
                if self.resource_prob is not None:
                    prob = self.resource_prob[self.season_idx]
                if not self.exists_type(type):
                    # replenish resource if gone
                    place_prob = 1
                elif self.resource_prob_decay and type in self.resource_prob_decay:
                    place_prob = max(0, prob - self.resource_prob_decay[type] * self.step_count)
                elif counts[type] > (self.grid_size - 2) ** 2 // max(8, len(self.seasons[self.season_idx])):
                    # don't add more if it's already taking up over 1/8 of the space (lower threshold if >10 diff obj types being generated)
                    place_prob = 0
                else:
                    place_prob = prob
                self.place_prob(TYPE_TO_CLASS_ABS[type](lifespan=self.lifespan), place_prob)

    def extra_gen_grid(self):
        for type, count in self.init_resources.items():
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
        self.just_eaten_type = None
        self.just_made_obj_type = None
        self.just_placed_on = None
        obs, reward, done, info = super().step(action)
        shelf_obs = self.gen_shelf_obs()

        # update seasons
        self.season_steps += 1
        if self.season_steps == self.season_lens[self.season_idx]:
            if self.season_idx == len(self.seasons) - 1:
                if self.reset_free:
                    self.season_idx = 0
                    self.season_steps = 0
                else:
                    done = True
            else:
                self.season_idx += 1
                self.season_steps = 0

        """ Generate obs """
        extra_obs = shelf_obs.flatten()
        # magic number repeating shelf 8 times to fill up more of the obs
        extra_obs = np.repeat(extra_obs, 8)
        obs = np.concatenate((obs, extra_obs))

        """ Generate reward """
        reward = self.reward()

        """ Generate info """
        info.update({'health': self.health})
        info.update(self.info_last)

        return obs, reward, done, info

    def reward(self):
        if self.rtype == 'health':
            return self.health
        elif self.rtype in ['waypoint', 'dense']:
            reward = 0
            make_sequence = self.season_seq[self.season_idx]
            carry_idx = make_sequence.index(
                self.carrying.type) if self.carrying and self.carrying.type in make_sequence else -1
            just_place_idx = make_sequence.index(
                self.just_placed_on.type) if self.just_placed_on and self.just_placed_on.type in make_sequence else -1
            made_idx = make_sequence.index(
                self.just_made_obj_type) if self.just_made_obj_type in make_sequence else -1
            just_eaten_idx = make_sequence.index(self.just_eaten_type) if self.just_eaten_type in make_sequence else -1
            pickup_idx = max(carry_idx, just_eaten_idx)
            max_idx = max(pickup_idx, just_place_idx)

            neg_rwd_idx = max(max_idx, made_idx)
            if self.task_seq[self.season_idx] == 'pickup' and pickup_idx == len(make_sequence) - 1:
                # exponent reasoning: 3rd obj in list should yield 100, 5th yields 10000, etc.
                reward = POS_RWD ** (pickup_idx // 2)
                self.onetime_reward_sequence = [False for _ in range(len(make_sequence))]
                self.num_solves += 1
                # remove the created goal object
                self.carrying = None
                # reset idxs
                self.last_idx = -1
                self.max_make_idx = -1
                return reward
            elif self.task_seq[self.season_idx] == 'make' and made_idx == len(make_sequence) - 1:
                # exponent reasoning: 3rd obj in list should yield 100, 5th yields 10000, etc.
                reward = POS_RWD ** (made_idx // 2)
                self.onetime_reward_sequence = [False for _ in range(len(make_sequence))]
                self.num_solves += 1
                # reset idxs
                self.last_idx = -1
                self.max_make_idx = -1
                return reward
            elif max_idx != -1 and not self.onetime_reward_sequence[max_idx]:
                # exponent reasoning: 3rd obj in list should yield 100, 5th yields 10000, etc.
                reward = POS_RWD ** (max_idx // 2)
                self.onetime_reward_sequence[max_idx] = True
            elif neg_rwd_idx < self.last_idx:
                reward = -np.abs(NEG_RWD ** (self.last_idx // 2 + 1))
            elif self.rtype == 'dense':
                next_pos = self.get_closest_obj_pos(make_sequence[max_idx + 1])
                if next_pos is not None:
                    dist = np.linalg.norm(next_pos - self.agent_pos, ord=1)
                    reward = -0.01 * dist

            if max_idx > self.max_make_idx:
                self.max_make_idx = max_idx

            self.last_idx = max_idx
            return reward

    def reset(self):
        if self.fixed_reset:
            self.seed(self.seed_val)
        obs = super().reset()
        extra_obs = np.repeat(self.gen_shelf_obs(), 8)
        obs = np.concatenate((obs, extra_obs.flatten()))

        self.pantry = []
        self.made_obj_type = None
        self.last_placed_on = None
        self.max_make_idx = -1
        self.last_idx = -1
        self.obs_count = {}
        self.inventory = self.gen_shelf_obs()

        self.onetime_reward_sequence = [False for _ in range(len(self.season_seq[self.season_idx]))]

        # Seasons
        self.season_idx = 0
        self.season_steps = 0

        return obs

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

    def decay_health(self):
        super().decay_health()


register(id='MiniGrid-Seasons-v0',
         entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:Seasons',
         kwargs=dict(
             grid_size=12,
             seasons=[
                         {
                             'sun': 0.04,
                             'seed': 0.04,
                             'water': 0.04
                         },
                         {
                             'metal': 0.04,
                             'energy': 0.04,
                             'tree': 0.02
                         }
                     ] * 2,
             season_lens=[200] * 4,
             season_seq=[
                            ['seed', 'water', 'plant', 'sun', 'tree'],
                            ['energy', 'metal', 'axe', 'tree', 'bigfood']
                        ] * 2,
             task_seq=['make', 'pickup'] * 2,
             lifespan=100,
             health_cap=1000,
             reset_free=False,
             rtype='dense',
             agent_view_size=9,
             resource_prob=[0.02, 0.02, 0.02, 0.02]
         )
         )
