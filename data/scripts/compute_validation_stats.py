import math
import os
import json
import argparse
import pickle
from datetime import datetime
from glob import glob
from collections import OrderedDict
from os.path import join, abspath, dirname, basename, normpath


def get_task_obj(variant_path):
    with open(variant_path, 'r') as f:
        variant = json.load(f)
        task = variant['env_kwargs']['task']
        return task.split()[1]


def compute_stats(stats, horizon, task_obj):
    """
    Computes the following stats:
        *   Proportion of envs solved
        *   Average solve time among solved envs
        *   Proportion of envs in which final obj made
    :param stats: objects that were pickled during exp in files of name format 'stats_<epoch>.pkl' containing
                    time indices of production and procurement of each type of object
    :param horizon: time horizon over which the stats are to be computed (only over the first HORIZON steps of stats)
                    If 0, then computes over full stats.
    :param task_obj: the type of object the agent was aiming to acquire for the exp (e.g. 'berry', 'axe', 'food')
    :return:
    """
    horizon = horizon or math.inf
    pickup_key = 'pickup_%s' % task_obj
    made_key = 'made_%s' % task_obj
    made_axe_key = 'made_axe'
    num_solves = 0
    # include possibly multiple solves per env
    num_solves_total = 0
    num_made = 0
    num_made_axe = 0
    solve_times = []
    for env_idx, stat in enumerate(stats):
        if pickup_key in stat and stat[pickup_key] and (stat[pickup_key][0] < horizon or horizon == 0):
            num_solves += 1
            num_solves_total += len(stat[pickup_key])
            solve_times.append(stat[pickup_key][0])
        if made_key in stat and stat[made_key] and (stat[made_key][0] < horizon or horizon == 0):
            num_made += 1
        if made_axe_key in stat and stat[made_axe_key] and (stat[made_axe_key][0] < horizon or horizon == 0):
            num_made_axe += 1
    proportion_solved = num_solves / len(stats)
    # can be greater than 1 if on avg solved >1 times per env
    proportion_solved_total = num_solves_total / len(stats)
    average_solve_time = sum(solve_times) / len(solve_times) if solve_times else 0
    proportion_made = num_made / len(stats)
    proportion_made_axe = num_made_axe / len(stats)

    return {
        'proportion_solved': proportion_solved,
        'proportion_solved_total': proportion_solved_total,
        'average_solve_time': average_solve_time,
        'proportion_made': proportion_made,
        'proportion_made_axe': proportion_made_axe
    }


def get_hash(variant):
    """
    Return unique identifier of variant env_kwargs and algo_kwargs
    """
    def dict_hash_recursive(dct):
        keys = sorted(dct.keys())
        hash_list = []
        for key in keys:
            val = dct[key]
            if type(val) is dict:
                hash_list.append(dict_hash_recursive(val))
            elif type(val) is list:
                hash_list.append(list_hash_recursive(val))
            else:
                hash_list.append(hash(str(val)))
        return hash(tuple(hash_list))

    def list_hash_recursive(lst):
        hash_list = []
        for val in lst:
            if type(val) is dict:
                hash_list.append(dict_hash_recursive(val))
            elif type(val) is list:
                hash_list.append(list_hash_recursive(val))
            else:
                hash_list.append(hash(str(val)))
        return hash(tuple(hash_list))

    return hash((
        dict_hash_recursive(variant['env_kwargs']),
        dict_hash_recursive(variant['algo_kwargs'])
    ))


def compute_validation_stats(exps_dir, horizon):
    """
    pseudocode
    ===========================
    make dict A of variant-hash to variant. to be saved for reference
    make dict B of variant-hash to a list, which will eventually have 3 elements
            corresponding to the stats for the 3 directories associated with that variant.
    get task from one of the exp_dir (e.g. 'make_lifelong axe')
    for each exp_dir in exps_dir:
        for horizon value and task, compute:
              percentage solved,
              average solve time among those solved,
              percentage where relevant obj is made (not just picked up),
              etc.
        append stats to B[hash(exp_dir.variant)]
    pickle both dicts together as final result
    ===========================
    result can be thrown into an ipynb to be plotted and whatnot
    """
    hash_to_variant = {}
    hash_to_stats = {}
    hash_to_seeds = {}

    dirpath, dirnames, _ = next(os.walk(exps_dir))

    variant_path = join(dirpath, dirnames[0], 'variant.json')
    # str representing the goal object, such as 'axe' or 'berry' or 'food'
    task_obj = get_task_obj(variant_path)

    # get immediate subdirectories. see https://stackoverflow.com/questions/800197/how-to-get-all-of-the-immediate-subdirectories-in-python
    for exp_dir in dirnames:
        full_path = join(dirpath, exp_dir)
        with open(join(full_path, 'variant.json'), 'r') as f:
            variant = json.load(f)
        variant_hash = get_hash(variant)
        hash_to_variant[variant_hash] = {k: variant[k] for k in ['env_kwargs', 'algo_kwargs']}
        hash_to_seeds[variant_hash] = variant['seed']
        if not hash_to_stats.get(variant_hash, False):
            hash_to_stats[variant_hash] = []
        # NOTE: this next line is hardcoded for stats files of format 'stats_<epoch>.pkl', eg 'stats_450.pkl'
        # lists stats_<epoch>.pkl files in increasing order of epoch
        epoch_to_stats = OrderedDict()
        for stats_file in sorted(glob(join(full_path, 'stats_*.pkl')), key=lambda f: int(basename(f)[6:-4])):
            epoch = int(basename(stats_file)[6:-4])
            with open(stats_file, 'rb') as f:
                stats = pickle.load(f)
            epoch_to_stats[epoch] = compute_stats(stats, horizon, task_obj)
        hash_to_stats[variant_hash].append(epoch_to_stats)
    return {
        'hash_to_variant': hash_to_variant,
        'hash_to_stats': hash_to_stats
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='generate processed stats from validation envs. based on the validation stats generated in `rl_algorithm.py`.')
    parser.add_argument('dir', help='full path to dir of exp dirs')
    parser.add_argument('--horizon', type=int, default=0, help='time horizon for stat computation')
    args = parser.parse_args()
    base_expdir = basename(normpath(args.dir))

    now = datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    cur_dir = dirname(abspath(__file__))
    validation_stats = compute_validation_stats(args.dir, args.horizon)
    save_dir = join(cur_dir, 'validation_stats', base_expdir)
    os.makedirs(save_dir, exist_ok=True)
    pickle.dump(validation_stats, open(join(save_dir, 'stats_%s.pkl' % timestamp), 'wb'))
