import math
import os
import pickle
import datetime

from gym_minigrid.envs.deer_diverse import DeerDiverseEnv
import numpy as np
import json
from gym_minigrid.envs.monsters import MonstersEnv


def gen_validation_envs(n, filename, **kwargs):
    envs = []
    seeds = np.random.randint(0, 100000, n).tolist()
    for idx in range(n):
        env_kwargs = dict(
            # sweep this
            deer_move_prob=0.2,
            # shaping params
            deer_dists=[{'easy': 0, 'medium': 0, 'hard': 1}, {'easy': 0, 'medium': 0, 'hard': 1}],
            # shaping period param (doesn't matter here  since start and end dists are the same)
            deer_dist_period=1,
            grid_size=10,
            agent_start_pos=None,
            health_cap=1000,
            gen_resources=True,
            fully_observed=False,
            task='make food',
            make_rtype='sparse',
            fixed_reset=False,
            only_partial_obs=True,
            init_resources={
                'axe': 2,
                'deer': 2
            },
            default_lifespan=0,
            fixed_expected_resources=True,
            end_on_task_completion=False,
            time_horizon=0,
            replenish_low_resources={
                'axe': 2,
                'deer': 2
            },
            seed=seeds[idx]
        )
        env_kwargs.update(**kwargs)
        env = DeerDiverseEnv(
            **env_kwargs
        )
        envs.append(env)
    pickle.dump({'envs': envs, 'seeds': seeds}, open(filename, 'wb'))
    json.dump(env_kwargs, open(filename.strip('.pkl') + '.json', 'w'),
              indent=4, sort_keys=True)
    print('Generated %d envs at file: %s' % (n, filename))


if __name__ == '__main__':
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    validation_dir = os.path.join(cur_dir, 'validation_envs')
    os.makedirs(validation_dir, exist_ok=True)

    filename = 'env_shaping_validation_envs_%s.pkl' % timestamp

    gen_validation_envs(100, os.path.join(validation_dir, filename))
