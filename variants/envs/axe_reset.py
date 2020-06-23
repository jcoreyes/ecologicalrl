env_variant = dict(
    grid_size=8,
    # start agent at random pos
    agent_start_pos=None,
    health_cap=1000,
    gen_resources=False,
    fully_observed=False,
    task='make axe',
    make_rtype='one-time',
    fixed_reset=False,
    only_partial_obs=True,
    init_resources={
        'metal': 1,
        'energy': 1
    },
    resource_prob={
        'metal': 0.01,
        'energy': 0.01
    },
    fixed_expected_resources=False,
    end_on_task_completion=False,
    time_horizon=250
)
