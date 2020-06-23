import copy
from variants.envs.axe_reset_free import env_variant

# prob not necessary but just to be safe since dicts are mutable
env_search_space = copy.deepcopy(env_variant)
# wrap values in list for sweeper
env_search_space = {k: [v] for k, v in env_search_space.items()}

# the sweep we want
env_search_space.update(
    resource_prob=[
        {'metal': 0.005, 'energy': 0.005},
        {'metal': 0.01, 'energy': 0.01},
        {'metal': 0.02, 'energy': 0.02},
        {'metal': 0.05, 'energy': 0.05}
    ],
    init_resources=[
        {'metal': 1, 'energy': 1},
        {'metal': 2, 'energy': 2}
    ],
    replenish_empty_resources=[True, False]
)




"""
9/4 03:00  : cache of version of env_variant above is meant to extend, in case it changes.
env_variant = dict(
    grid_size=8,
    # start agent at random pos
    agent_start_pos=None,
    health_cap=1000,
    gen_resources=True,
    fully_observed=False,
    task='make_lifelong axe',
    make_rtype='sparse',
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
    fixed_expected_resources=False,
    end_on_task_completion=False,
    time_horizon=0
)

"""