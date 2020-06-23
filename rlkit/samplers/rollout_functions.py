import time

# from array2gif import write_gif
import numpy as np
from PIL import Image
import os
from os.path import join

def multitask_rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        observation_key=None,
        desired_goal_key=None,
        get_action_kwargs=None,
        return_dict_obs=False,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    dict_obs = []
    dict_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if render:
        env.render(**render_kwargs)
        time.sleep(0.25)
    goal = o[desired_goal_key]
    while path_length < max_path_length:
        dict_obs.append(o)
        if observation_key:
            o = o[observation_key]
        new_obs = np.hstack((o, goal))
        a, agent_info = agent.get_action(new_obs, **get_action_kwargs)
        next_o, r, d, env_info = env.step(a)
        if render:
            env.render(**render_kwargs)
            time.sleep(0.1)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        dict_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = dict_obs
        next_observations = dict_next_obs
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        goals=np.repeat(goal[None], path_length, 0),
        full_observations=dict_obs,
    )


def hierarchical_rollout(
        env,
        agent,
        setter,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        return_env_obs=False,
        continuing=False,
        obs=None
):
    if continuing:
        assert obs is not None, 'if continuing, then must provide the most recent obs and goal'

    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    dict_obs = []
    dict_next_obs = []
    observations = []
    goals = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0

    high_observations = []
    high_actions = []
    high_rewards = []
    high_terminals = []
    high_infos = []
    high_env_infos = []
    high_next_observations = []
    # low level transitions comprising high level action (each goal)
    high_traj_obs = []
    high_traj_acs = []
    last_high_traj_idx = 0
    high_path_length = 0
    # returns btwn high level actions
    high_reward = 0

    if continuing:
        o = obs
    else:
        agent.reset()
        setter.reset()
        o = env.reset()
    if render:
        env.render(**render_kwargs)
        time.sleep(0.25)
    while path_length < max_path_length:
        dict_obs.append(o)
        g, high_info = setter.get_action(o)
        new_obs = np.hstack((o, g))
        a, agent_info = agent.get_action(new_obs, **get_action_kwargs)
        next_o, r, d, env_info = env.step(a)
        if render:
            env.render(**render_kwargs)
            time.sleep(0.1)
        if path_length and high_info['new_goal']:
            high_observations.append(o)
            high_actions.append(g)
            high_rewards.append(high_reward)
            high_reward = 0
            high_terminals.append(d)
            # not next_o since next_obs was reached as a result of the new goal
            high_next_observations.append(o)
            high_infos.append(high_info)
            high_env_infos.append(env_infos)
            high_traj_obs.append(observations[last_high_traj_idx:])
            high_traj_acs.append(actions[last_high_traj_idx:])
            last_high_traj_idx = len(observations)
            high_path_length += 1
        observations.append(o)
        goals.append(g)
        rewards.append(setter.get_reward(o, g, a, next_o))
        high_reward += r
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        dict_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        if path_length == max_path_length - 1:
            high_observations.append(o)
            high_actions.append(g)
            high_rewards.append(high_reward)
            high_reward = 0
            high_terminals.append(d)
            # not next_o since next_obs was reached as a result of the new goal
            high_next_observations.append(o)
            high_infos.append(high_info)
            high_env_infos.append(env_infos)
            high_traj_obs.append(observations[last_high_traj_idx:])
            high_traj_acs.append(actions[last_high_traj_idx:])
            last_high_traj_idx = len(observations)
            high_path_length += 1
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    high_actions = np.array(high_actions)
    goals = np.array(goals)
    if len(goals.shape) == 1:
        goals = np.expand_dims(goals, 1)
    high_observations = np.array(high_observations)
    high_next_observations = np.array(high_next_observations)
    high_traj_obs = np.array(high_traj_obs)
    high_traj_acs = np.array(high_traj_acs)

    if return_dict_obs:
        observations = dict_obs
        next_observations = dict_next_obs
    low_path = dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        goals=goals,
        full_observations=dict_obs,
    )
    high_path = dict(
        observations=high_observations,
        actions=high_actions,
        rewards=np.array(high_rewards).reshape(-1, 1),
        next_observations=high_next_observations,
        terminals=np.array(high_terminals).reshape(-1, 1),
        traj_obs=high_traj_obs,
        traj_acs=high_traj_acs,
        agent_infos=high_infos,
        env_infos=env_infos,
        goals=goals,
        full_observations=dict_obs,
    )

    return (low_path, high_path, env, o) if return_env_obs else (low_path, high_path)


def rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        return_env_obs=False,
        continuing=False,
        obs=None,
        save=False,
        save_dir=None
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    If `return_env_obs` is True, then return the env and last obs as well.
    If `continuing` is True, then roll out without resetting env. `obs` must then be the most recent obs from the env
    """
    assert not (continuing and obs is None), 'if continuing, then must provide the most recent obs from the env'
    assert not (save and save_dir is None), 'if saving, must provide dir to save to'

    def save_img(img, path):
        img = img.getArray()
        im = Image.fromarray(img)
        im.save(path)
        return img.transpose(1, 0, 2)

    if save:
        img_dir = join(save_dir, 'imgs')
        os.makedirs(img_dir, exist_ok=True)

    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []

    imgs = []
    if continuing:
        o = obs
    else:
        o = env.reset()
        agent.reset()
    next_o = None
    path_length = 0
    if render:
        img = env.render(**render_kwargs)
        if save:
            img = save_img(img, join(img_dir, '%d.png' % path_length))
            imgs.append(img)
        time.sleep(0.25)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        # if render:
        #     print(a)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        o = next_o
        if render:
            img = env.render(**render_kwargs)
            if save:
                img = save_img(img, join(img_dir, '%d.png' % path_length))
                imgs.append(img)
            time.sleep(0.1)
        if d:
            break
    if save:
        write_gif(imgs, join(save_dir, 'full.gif'), fps=5)
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    if type(observations[0]) is not np.array:
        observations = [x.__array__() for x in observations]
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    ret = dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )
    return (ret, env, next_o) if return_env_obs else ret


def rollout_config(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        return_env_obs=False,
        continuing=False,
        obs=None,
        seed=None
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    If `return_env_obs` is True, then return the env and last obs as well.
    If `continuing` is True, then roll out without resetting env. `obs` must then be the most recent obs from the env
    """
    assert not (continuing and (obs is None or seed is None)), \
        'if continuing, then must provide the most recent obs and seed from the env'

    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    if continuing:
        o = obs
    else:
        o, seed = env.reset(seed=seed, return_seed=True)
        agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
        time.sleep(0.25)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        # if render:
        #     print(a)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        o = next_o
        if render:
            env.render(**render_kwargs)
            time.sleep(0.1)
        if d:
            break
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    if type(observations[0]) is not np.array:
        observations = [x.__array__() for x in observations]
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    ret = dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )
    return (ret, env, next_o, seed) if return_env_obs else (ret, seed)


def env_shape_rollout(
        env,
        eval_env,
        agent,
        parent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        return_env_obs=False,
        continuing=False,
        obs=None
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    If `return_env_obs` is True, then return the env and last obs as well.
    If `continuing` is True, then roll out without resetting env. `obs` must then be the most recent obs from the env

    Eval_env is used to provide reward to the parent
    """
    assert not (continuing and obs is None), 'if continuing, then must provide the most recent obs from the env'

    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    if continuing:
        o = obs
    else:
        o = env.reset()
        agent.reset()
        parent.reset()
    next_o = None
    path_length = 0
    num_trajs = 0


    params = parent.get_action(o)
    if render:
        env.render(**render_kwargs)
        time.sleep(0.25)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        o = next_o
        if render:
            env.render(**render_kwargs)
            time.sleep(0.1)
        if d:
            # break
            o = env.reset()
            agent.reset()
            num_trajs += 1
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    ret = dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )
    return (ret, env, next_o) if return_env_obs else ret