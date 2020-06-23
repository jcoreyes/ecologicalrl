import os
import pickle
import datetime

from gym_minigrid.envs.deer import DeerEnv
import numpy as np
from gym_minigrid.envs.tools import ToolsEnv
import json

def gen_validation_envs(n, filename, **kwargs):
    envs = []
    seeds = np.random.randint(0, 100000, n).tolist()
    for idx in range(n):
        env_kwargs = dict(
            grid_size=8,
            # start agent at random pos
            agent_start_pos=None,
            health_cap=1000,
            gen_resources=True,
            fully_observed=False,
            task='make food',
            make_rtype='sparse',
            fixed_reset=False,
            only_partial_obs=True,
            init_resources={
                'axe': 1,
                'deer': 1
            },
            deer_move_prob=0.1,
            fixed_expected_resources=True,
            end_on_task_completion=True,
            time_horizon=100,
            seed=seeds[idx]
        )
        env_kwargs.update(**kwargs)
        env = DeerEnv(
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

    filename = 'dynamic_static_validation_envs_%s.pkl' % timestamp

    gen_validation_envs(1, os.path.join(validation_dir, filename))
