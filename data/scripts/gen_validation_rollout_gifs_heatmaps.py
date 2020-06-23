import argparse
import os
from os.path import join, basename, isfile
import time
import pathlib

from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import math
from itertools import chain
import json
import pickle
import random
from rlkit.core.logging import get_repo_dir
from array2gif import write_gif
from PIL import Image
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedyDecay, EpsilonGreedy
from rlkit.samplers.rollout_functions import rollout


def get_val_envs(val_envs_path):
    print('OVERRIDING VAL ENVS')
    val_envs_path = 'VAL_ENVS_PATH'
    with open(val_envs_path, 'rb') as f:
        envs = pickle.load(f)['envs']
    return envs


def get_gifs_heatmaps(exps_dir_name, seeds, save_dir, titles):
    data_dir = join(get_repo_dir(), 'data')
    exps_dir = join(data_dir, exps_dir_name)
    gifs_dir = join(data_dir, 'gifs')
    heat_dir = join(data_dir, 'heatmaps')

    # load variant and get pickled validation envs
    rand_exp_dir = glob(join(exps_dir, '*'))[0]
    with open(join(rand_exp_dir, 'variant.json'), 'r') as f:
        variant = json.load(f)
    task_obj = variant['env_kwargs']['task'].split()[1]
    val_envs_path = variant['algo_kwargs']['algorithm_kwargs']['validation_envs_pkl']
    val_rollout_len = variant['algo_kwargs']['algorithm_kwargs']['validation_rollout_length']
    val_envs = get_val_envs(val_envs_path)

    # load policy
    for seed_idx, seed in enumerate(seeds):
        val_env_idxs = random.sample(list(range(len(val_envs))), 10)
        exp_dir = glob(join(exps_dir, '*%d' % seed))[0]

        """ Get policy """
        pol_file = max(glob(join(exp_dir, 'itr_*.pkl')), key=lambda pol_path: int(basename(pol_path)[4:-4]))
        # to override policy itr number
        # pol_file = join(exp_dir, 'itr_%d.pkl' % 2990)
        print(pol_file)
        with open(pol_file, 'rb') as f:
            policy = pickle.load(f)['evaluation/policy']
        if hasattr(policy, 'policy'):
            # if it's reset free, strip out the underlying policy from the exploration strategy
            policy = policy.policy
        policy = PolicyWrappedWithExplorationStrategy(
            EpsilonGreedy(spaces.Discrete(7), 0.1),
            policy
        )

        # re-fetch the val envs each time so that envs are fresh
        # val_envs = get_val_envs(val_envs_path)
        # """ Get gifs """
        # stats = [{} for _ in range(len(val_env_idxs))]
        # for meta_idx, env_idx in enumerate(val_env_idxs):
        #     env = val_envs[env_idx]
        #     path = rollout(env, policy, val_rollout_len, render=True, save=True,
        #                    save_dir=join(gifs_dir, exps_dir_name, save_dir, str(seed), str(env_idx)))
        #     env.render(close=True)
        #     for typ in env.object_to_idx.keys():
        #         if typ not in ['empty', 'wall', 'tree']:
        #             key = 'pickup_%s' % typ
        #             last_val = 0
        #             pickup_idxs = []
        #             for t, env_info in enumerate(path['env_infos']):
        #                 count = env_info[key] - last_val
        #                 pickup_idxs.extend([t for _ in range(count)])
        #                 last_val = env_info[key]
        #             stats[meta_idx][key] = pickup_idxs
        #     for typ in env.interactions.values():
        #         key = 'made_%s' % typ
        #         last_val = 0
        #         made_idxs = []
        #         for t, env_info in enumerate(path['env_infos']):
        #             count = env_info[key] - last_val
        #             made_idxs.extend([t for _ in range(count)])
        #             last_val = env_info[key]
        #         stats[meta_idx][key] = made_idxs
        # solved = [val_env_idxs[i] for i, stat in enumerate(stats) if stat['pickup_%s' % task_obj]]
        # print('seed %d solved %d percent:' % (seed, 100 * len(solved) // len(val_env_idxs)))
        # print(solved)

        # re-fetch the val envs each time so that envs are fresh
        val_envs = get_val_envs(val_envs_path)
        print('refetched envs')
        """ Get heatmaps """
        vcs = []
        for env_idx, env in enumerate(val_envs):
            path = rollout(env, policy, val_rollout_len)
            vcs.append(env.visit_count)
        visit_count_sum = sum(vcs)
        plt.imshow(visit_count_sum)
        plt.title('Validation Tasks State Visitation Count (%s)' % titles[seed_idx])
        plt.axis('off')
        vc_save_path = join(heat_dir, exps_dir_name, save_dir, str(seed))
        os.makedirs(vc_save_path, exist_ok=True)
        plt.savefig(join(vc_save_path, 'map.png'))



if __name__ == '__main__':
    # exps_dir_name = '09-19-17-tool-dqn-dynamic-static-resetfree'
    # seeds = [14186]#, 49761, 16103]
    # # titles of exps in positions corresponding to those in `seeds`
    # titles = ['One-time reward shaping']#, 'One-time reward shaping', 'Environment shaping']
    # save_dir = 'deer_shaping'
    # get_gifs_heatmaps(exps_dir_name, seeds, save_dir, titles)

    parser = argparse.ArgumentParser(
        description='generate rollout gifs and heatmaps from running the policies in validation envs')
    parser.add_argument('dir', help='full path to dir of exp dirs')
    parser.add_argument('seed', type=int, help='seed specifying which exp dir to use')
    parser.add_argument('save_dir', help='an identifier that the save dir will be named after')
    parser.add_argument('title', help='the title to be put on the generated heatmap (e.g. "Distance-based reward shaping")')
    args = parser.parse_args()
    get_gifs_heatmaps(args.dir, [args.seed], args.save_dir, [args.title])
