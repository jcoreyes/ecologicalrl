#!/usr/bin/env python3

from __future__ import division, print_function

import math
import pickle
import sys

from gym_minigrid.envs import FoodEnvHard1Inv
from gym_minigrid.envs.deer import DeerEnv
from gym_minigrid.envs.deer_diverse import DeerDiverseEnv
from gym_minigrid.envs.factory import FactoryEnv
from gym_minigrid.envs.lava import LavaEnv
from gym_minigrid.envs.monsters import MonstersEnv
from gym_minigrid.envs.tools import ToolsEnv, ToolsWallEnv
import numpy
import gym
import time
from optparse import OptionParser

import gym_minigrid
from rlkit.torch.core import torch_ify


def main():
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-MultiRoom-N6-v0'
    )
    parser.add_option(
        "-q",
        "--qfunc",
        dest="qf",
        help="path to pickle file of q network to load",
        default=None
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = FactoryEnv(
        fac_move_prob=0.3,
        fac_move_close_prob=0.8,
        fac_move_close_prob_decay=0.01,
        grid_size=8,
        # start agent at random pos
        agent_start_pos=None,
        health_cap=1000,
        gen_resources=False,
        fully_observed=False,
        task='make lava',
        make_rtype='dense-fixed',
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
        make_sequence=['metal', 'wood', 'axe', 'lava']
    )
    # env = DeerDiverseEnv(
    #     # sweep this
    #     deer_move_prob=0.5,
    #     # shaping params (dynamism just has med throughout, with diff deer move probs)
    #     deer_dists=[{'easy': 1, 'medium': 0, 'hard': 0}, {'easy': 0, 'medium': 0.2, 'hard': 0.8}],
    #     # shaping period param
    #     deer_dist_period=10,
    #     grid_size=10,
    #     agent_start_pos=None,
    #     health_cap=1000,
    #     gen_resources=True,
    #     fully_observed=False,
    #     task='make food',
    #     make_rtype='dense-fixed',
    #     fixed_reset=False,
    #     only_partial_obs=True,
    #     init_resources={
    #         # 'metal': 1,
    #         # 'wood': 1
    #         'axe': 2,
    #         'deer': 2
    #     },
    #     default_lifespan=0,
    #     fixed_expected_resources=True,
    #     end_on_task_completion=False,
    #     time_horizon=0,
    #     replenish_low_resources={
    #         'axe': 2,
    #         'deer': 2
    #     },
    # )
    # env = MonstersEnv(
    #     monster_eps=0,
    #     monster_attack_dist=0,
    #     grid_size=10,
    #     # start agent at random pos
    #     agent_start_pos=None,
    #     health_cap=1000,
    #     gen_resources=True,
    #     fully_observed=False,
    #     task='make food',
    #     make_rtype='sparse',
    #     fixed_reset=False,
    #     only_partial_obs=True,
    #     init_resources={
    #         'food': 2,
    #         'monster': 2
    #     },
    #     resource_prob={
    #         'food': 0.1,
    #         'monster': 0.0
    #     },
    #     lifespans={
    #         'monster': 20,
    #     },
    #     replenish_low_resources={
    #         'food': 2,
    #         'monster': 2
    #     },
    #     place_schedule=(50, 25),
    #     fixed_expected_resources=True,
    #     end_on_task_completion=True,
    #     time_horizon=0
    # )
    # env = LavaEnv(
    #     num_lava=5,
    #     lava_timeout=1,
    #     lava_penalty=10,
    #     grid_size=10,
    #     agent_start_pos=None,
    #     health_cap=1000,
    #     gen_resources=True,
    #     fully_observed=False,
    #     task='make axe',
    #     make_rtype='sparse',
    #     fixed_reset=False,
    #     only_partial_obs=True,
    #     init_resources={
    #         'metal': 2,
    #         'wood': 2
    #     },
    #     replenish_low_resources={
    #         'metal': 2,
    #         'wood': 2
    #     },
    #     end_on_task_completion=False,
    #     # num_walls=3,
    #     # fixed_walls=True,
    #     time_horizon=0,
    #     agent_view_size=5,
    # )
    # env = gym.make(options.env_name)
    pkl = options.qf
    if pkl is not None:
        params = pickle.load(open(pkl, 'rb'))
        qf = params['trainer/qf']

    def resetEnv():
        env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)

    resetEnv()

    # Create a window to render into
    renderer = env.render('human')

    def keyDownCb(keyName):
        if keyName == 'BACKSPACE':
            resetEnv()
            return

        if keyName == 'ESCAPE':
            sys.exit(0)

        action = 0

        if keyName == 'LEFT':
            action = env.actions.west
        elif keyName == 'RIGHT':
            action = env.actions.east
        elif keyName == 'UP':
            action = env.actions.north
        elif keyName == 'DOWN':
            action = env.actions.south

        elif keyName == 'SPACE':
            action = env.actions.mine
        elif keyName == 'PAGE_UP':
            if hasattr(env.actions, 'eat'):
                action = env.actions.eat
            else:
                action = env.actions.dispense
        elif keyName == 'PAGE_DOWN':
            action = env.actions.place
        elif keyName == '0':
            action = env.actions.place0
        elif keyName == '1':
            action = env.actions.place1
        elif keyName == '2':
            action = env.actions.place2
        elif keyName == '3':
            action = env.actions.place3
        elif keyName == '4':
            action = env.actions.place4
        elif keyName == 'RETURN':
            action = env.actions.done

        else:
            print("unknown key %s" % keyName)
            return

        obs, reward, done, info = env.step(action)
        if pkl is not None:
            qs = qf(torch_ify(obs)).data.numpy()[0]
            print(qs)
            print(qs.argmax())
        if hasattr(env, 'health'):
            print('step=%s, reward=%.2f, health=%d' % (env.step_count, reward, env.health))
        else:
            print('step=%s, reward=%.2f' % (env.step_count, reward))
        if done:
            print('done!')
            resetEnv()

    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human')
        time.sleep(0.01)

        # If the window was closed
        if renderer.window == None:
            break


if __name__ == "__main__":
    main()
