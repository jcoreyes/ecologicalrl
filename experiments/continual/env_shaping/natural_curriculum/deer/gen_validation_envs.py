import os
import pickle
import datetime

from gym_minigrid.envs.deer import DeerEnv
import numpy as np
from gym_minigrid.envs.tools import ToolsEnv


def gen_validation_envs(n, filename, **kwargs):
    envs = []
    seeds = np.random.randint(0, 100000, n).tolist()
    for idx in range(n):
        env_kwargs = dict(
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
                'metal': 3,
                'wood': 3,
                'deer': 3
            },
            default_lifespan=0,
            fixed_expected_resources=True,
            end_on_task_completion=False,
            time_horizon=1000,
            replenish_low_resources={
                'metal': 3,
                'wood': 3,
                'deer': 3
            },
            seed=seeds[idx]
        )
        env_kwargs.update(**kwargs)
        env = DeerEnv(
            **env_kwargs
        )
        envs.append(env)
    pickle.dump({'envs': envs, 'seeds': seeds}, open(filename, 'wb'))
    print('Generated %d envs at file: %s' % (n, filename))


if __name__ == '__main__':
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    validation_dir = os.path.join(cur_dir, 'validation_envs')
    os.makedirs(validation_dir, exist_ok=True)

    filename = 'dynamic_static_validation_envs_%s.pkl' % timestamp

    gen_validation_envs(100, os.path.join(validation_dir, filename))
