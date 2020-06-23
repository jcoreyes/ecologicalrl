import csv
import math
import os
import pickle
import argparse
from os.path import join, isfile

from gym_minigrid.minigrid_absolute import TYPE_TO_CLASS_ABS
from rlkit.core.logging import compute_stats, get_task_obj
from rlkit.samplers.rollout_functions import rollout


def validate(policy, envs, horizon):
    """
    Collect list of stats for each validation env as dict of following format:
        'pickup_wood': [0, 15, 20] means you picked up a wood object at timesteps 0, 15, and 20.
    """
    stats = [{} for _ in range(len(envs))]
    for env_idx, env in enumerate(envs):
        path = rollout(env, policy, horizon)
        for typ in env.object_to_idx.keys():
            if typ in TYPE_TO_CLASS_ABS and TYPE_TO_CLASS_ABS[typ]().can_mine(env):
                key = 'pickup_%s' % typ
                last_val = 0
                pickup_idxs = []
                for t, env_info in enumerate(path['env_infos']):
                    count = env_info[key] - last_val
                    pickup_idxs.extend([t for _ in range(count)])
                    last_val = env_info[key]
                stats[env_idx][key] = pickup_idxs
        for typ in env.interactions.values():
            key = 'made_%s' % typ
            last_val = 0
            made_idxs = []
            for t, env_info in enumerate(path['env_infos']):
                count = env_info[key] - last_val
                made_idxs.extend([t for _ in range(count)])
                last_val = env_info[key]
            stats[env_idx][key] = made_idxs
    return stats


def run_validation(exps_dir, envs, period, horizon, nowrite, suffix, override):
    """
    :param exps_dir: the exp dir with multiple individual runs for seeds within. policies pulled from the inner dirs
    :param envs: list of envs to roll out the policies in
    :param period: how often policies were saved. determines the files from which you pull the policies
    :param horizon: time horizon for rollouts
    :param nowrite: whether or not to write to pickle file. if true, below two params don't matter
    :param suffix: string to append to saved files of format 'stats_<suffix>_<itr>.pkl'
    :param override: whether it's ok to override stats files that exist already
    :return:
    """
    dirpath, dirnames, _ = next(os.walk(exps_dir))
    horizon = horizon or math.inf
    task_obj = get_task_obj(join(dirpath, dirnames[0], 'variant.json'))
    writer = None

    for exp_dir in dirnames:
        full_path = join(dirpath, exp_dir)
        itr = 0
        itr_path = join(full_path, 'itr_%d.pkl' % itr)
        stats_name = 'validation_stats_%s.csv' % suffix if suffix else 'validation_stats.csv'
        stats_path = join(full_path, stats_name)
        assert override or not isfile(stats_path), 'stats file already exists: %s' % stats_path
        with open(stats_path, 'w') as stat_file:
            while isfile(itr_path) and itr < 100:
                with open(itr_path, 'rb') as f:
                    policy = pickle.load(f)['evaluation/policy']
                stats = validate(policy, envs, horizon)
                save_stats = compute_stats(stats, horizon=0, task_obj=task_obj)
                if not nowrite:
                    if itr == 0:
                        # if it's the first loop, initialize csv writer and write header (can't do earlier because don't know field names
                        writer = csv.DictWriter(stat_file, fieldnames=list(sorted(save_stats.keys())))
                        writer.writeheader()
                    writer.writerow(save_stats)
                itr += period
                itr_path = join(full_path, 'itr_%d.pkl' % itr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Roll out saved policies on validation envs.')
    parser.add_argument('dir', help='full path to dir of exp dirs')
    parser.add_argument('envs', help='pickle file containing the validation envs')
    parser.add_argument('--period', type=int, default=10, help='period of pickled policies to roll out')
    parser.add_argument('--horizon', type=int, default=0, help='time horizon for rollout')
    parser.add_argument('--nowrite', action='store_true', help='if set, doesnt write results to pkl files')
    parser.add_argument('--suffix', help='the label appended to the stats file')
    parser.add_argument('--override', action='store_true')
    args = parser.parse_args()

    assert args.period > 0, 'must have positive period'

    with open(args.envs, 'rb') as f:
        envs = pickle.load(f)['envs']

    run_validation(args.dir, envs, args.period, args.horizon, args.nowrite, args.suffix, args.override)
