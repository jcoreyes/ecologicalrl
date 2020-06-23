import os
import pickle
import datetime

from gym_minigrid.envs.factory import FactoryEnv
import numpy as np
import random
from gym_minigrid.envs.tools import ToolsEnv
import json

def gen_validation_envs(n, filename, **kwargs):
    envs = []
    seeds = [random.randint(0, 100000) for _ in range(n)]
    for idx in range(n):
        env_kwargs = dict(
            grid_size=8,
            # start agent at random pos
            agent_start_pos=None,
            health_cap=1000,
            gen_resources=False,
            fully_observed=False,
            task='make lava',
            make_rtype='sparse',
            fixed_reset=False,
            only_partial_obs=True,
            init_resources={
                'metalfactory': 1,
                'woodfactory': 1,
                'lava': 1
            },
            resource_prob={
                'metal': 0.0,
                'wood': 0.0
            },
            fixed_expected_resources=True,
            end_on_task_completion=True,
            time_horizon=0,
            make_sequence=['metal', 'wood', 'axe', 'lava'],
            seed=seeds[idx]
        )
        env_kwargs.update(**kwargs)
        env = FactoryEnv(
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
