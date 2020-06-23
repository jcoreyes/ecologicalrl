import os
import pickle
import datetime
import numpy as np
from gym_minigrid.envs.tools import ToolsEnv
import json

def gen_validation_envs(n, filename, **kwargs):
    envs = []
    seeds = np.random.randint(0, 100000, n).tolist()
    for idx in range(n):
        env_kwargs = {
           "agent_start_pos": None,
           "end_on_task_completion": True,
           "fixed_expected_resources": True,
           "fixed_reset": False,
           "fully_observed": False,
           "gen_resources": False,
           "grid_size": 8,
           "health_cap": 1000,
           "init_resources": {
              "metal": 2,
              "wood": 2
           },
           "make_rtype": "sparse",
           "only_partial_obs": True,
           "resource_prob": {
              "metal": 0.0,
              "wood": 0.0
           },
           "task": "make axe",
           "time_horizon": 200
        }
        env_kwargs = dict(
            grid_size=8,
            # start agent at random pos
            agent_start_pos=None,
            health_cap=1000,
            gen_resources=False,
            fully_observed=False,
            task='make axe',
            make_rtype='sparse',
            fixed_reset=False,
            only_partial_obs=True,
            init_resources={
                'metal': 1,
                'wood': 1
            },
            resource_prob={
                'metal': 0.0,
                'wood': 0.0
            },
            fixed_expected_resources=True,
            end_on_task_completion=True,
            time_horizon=100,
            seed=seeds[idx]
        )
        env_kwargs.update(**kwargs)
        env = ToolsEnv(
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

    gen_validation_envs(100, os.path.join(validation_dir, filename))
