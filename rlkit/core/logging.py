"""
Based on rllab's logger.

https://github.com/rll/rllab
"""
from enum import Enum
from contextlib import contextmanager
import math
import torch

import numpy as np
# import matplotlib.pyplot as plt
import os.path as osp
import sys
import datetime
import dateutil.tz
import csv
import errno
import os
import json
import pickle
import datetime
from glob import glob
from collections import OrderedDict
from os.path import join, abspath, dirname, basename, normpath

from rlkit.core.tabulate import tabulate


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
        try:
            with open(join(full_path, 'variant.json'), 'r') as f:
                variant = json.load(f)
        except FileNotFoundError:
            print('Exp dir %s missing variant file. Skipping...' % exp_dir)
            continue
        variant_hash = get_hash(variant)
        hash_to_variant[variant_hash] = {k: variant[k] for k in ['env_kwargs', 'algo_kwargs']}
        hash_to_seeds[variant_hash] = variant['seed']
        if not hash_to_stats.get(variant_hash, False):
            hash_to_stats[variant_hash] = []
        # NOTE: this next line is hardcoded for stats files of format 'stats_<epoch>.pkl', eg 'stats_450.pkl'
        # lists stats_<epoch>.pkl files in increasing order of epoch
        epoch_to_stats = OrderedDict()
        stats_files = glob(join(full_path, 'stats_*.pkl'))
        if not stats_files:
            continue
        for stats_file in sorted(stats_files, key=lambda f: int(basename(f)[6:-4])):
            epoch = int(basename(stats_file)[6:-4])
            with open(stats_file, 'rb') as f:
                stats = pickle.load(f)
            epoch_to_stats[epoch] = compute_stats(stats, horizon, task_obj)
        epoch_stats_np = np.zeros((len(stats_files), 1 + len(epoch_to_stats[0])))  # + 1 because want a column for the time index
        for idx, (epoch, stats) in enumerate(epoch_to_stats.items()):
            # TODO   hardcoded timesteps per epoch as 500 :(
            row = [epoch * 500] + [stats[k] for k in sorted(stats.keys())]
            epoch_stats_np[idx] = np.array(row)
        columns = ['timestep'] + list(sorted(epoch_to_stats[0].keys()))
        save_path = join(exps_dir, dirpath, exp_dir, 'validation_stats.csv')
        np.savetxt(save_path, epoch_stats_np, delimiter=",", header=','.join(columns), comments='')
        hash_to_stats[variant_hash].append(epoch_to_stats)
    return {
        'hash_to_variant': hash_to_variant,
        'hash_to_stats': hash_to_stats
    }

class TerminalTablePrinter(object):
    def __init__(self):
        self.headers = None
        self.tabulars = []

    def print_tabular(self, new_tabular):
        if self.headers is None:
            self.headers = [x[0] for x in new_tabular]
        else:
            assert len(self.headers) == len(new_tabular)
        self.tabulars.append([x[1] for x in new_tabular])
        self.refresh()

    def refresh(self):
        import os
        rows, columns = os.popen('stty size', 'r').read().split()
        tabulars = self.tabulars[-(int(rows) - 3):]
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(tabulate(tabulars, self.headers))
        sys.stdout.write("\n")


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class Logger(object):
    def __init__(self):
        self._prefixes = []
        self._prefix_str = ''

        self._tabular_prefixes = []
        self._tabular_prefix_str = ''

        self._tabular = []

        self._text_outputs = []
        self._tabular_outputs = []

        self._text_fds = {}
        self._tabular_fds = {}
        self._tabular_header_written = set()

        self._snapshot_dir = None
        self._snapshot_mode = 'all'
        self._snapshot_gap = 1

        self._log_tabular_only = False
        self._header_printed = False
        self.table_printer = TerminalTablePrinter()

    def reset(self):
        self.__init__()

    def _add_output(self, file_name, arr, fds, mode='a'):
        if file_name not in arr:
            mkdir_p(os.path.dirname(file_name))
            arr.append(file_name)
            fds[file_name] = open(file_name, mode)

    def _remove_output(self, file_name, arr, fds):
        if file_name in arr:
            fds[file_name].close()
            del fds[file_name]
            arr.remove(file_name)

    def push_prefix(self, prefix):
        self._prefixes.append(prefix)
        self._prefix_str = ''.join(self._prefixes)

    def add_text_output(self, file_name):
        self._add_output(file_name, self._text_outputs, self._text_fds,
                         mode='a')

    def remove_text_output(self, file_name):
        self._remove_output(file_name, self._text_outputs, self._text_fds)

    def add_tabular_output(self, file_name, relative_to_snapshot_dir=False):
        if relative_to_snapshot_dir:
            file_name = osp.join(self._snapshot_dir, file_name)
        self._add_output(file_name, self._tabular_outputs, self._tabular_fds,
                         mode='w')

    def remove_tabular_output(self, file_name, relative_to_snapshot_dir=False):
        if relative_to_snapshot_dir:
            file_name = osp.join(self._snapshot_dir, file_name)
        if self._tabular_fds[file_name] in self._tabular_header_written:
            self._tabular_header_written.remove(self._tabular_fds[file_name])
        self._remove_output(file_name, self._tabular_outputs, self._tabular_fds)

    def set_snapshot_dir(self, dir_name):
        self._snapshot_dir = dir_name

    def get_snapshot_dir(self, ):
        return self._snapshot_dir

    def get_snapshot_mode(self, ):
        return self._snapshot_mode

    def set_snapshot_mode(self, mode):
        self._snapshot_mode = mode

    def get_snapshot_gap(self, ):
        return self._snapshot_gap

    def set_snapshot_gap(self, gap):
        self._snapshot_gap = gap

    def set_log_tabular_only(self, log_tabular_only):
        self._log_tabular_only = log_tabular_only

    def get_log_tabular_only(self, ):
        return self._log_tabular_only

    def log(self, s, with_prefix=True, with_timestamp=True):
        out = s
        if with_prefix:
            out = self._prefix_str + out
        if with_timestamp:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
            out = "%s | %s" % (timestamp, out)
        if not self._log_tabular_only:
            # Also log to stdout
            print(out)
            for fd in list(self._text_fds.values()):
                fd.write(out + '\n')
                fd.flush()
            sys.stdout.flush()

    def record_tabular(self, key, val):
        self._tabular.append((self._tabular_prefix_str + str(key), str(val)))

    def record_dict(self, d, prefix=None):
        if prefix is not None:
            self.push_tabular_prefix(prefix)
        for k, v in d.items():
            self.record_tabular(k, v)
        if prefix is not None:
            self.pop_tabular_prefix()

    def push_tabular_prefix(self, key):
        self._tabular_prefixes.append(key)
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def pop_tabular_prefix(self, ):
        del self._tabular_prefixes[-1]
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def save_extra_data(self, data, file_name='extra_data.pkl', mode='joblib'):
        """
        Data saved here will always override the last entry

        :param data: Something pickle'able.
        """
        file_name = osp.join(self._snapshot_dir, file_name)
        if mode == 'joblib':
            import joblib
            joblib.dump(data, file_name, compress=3)
        elif mode == 'pickle':
            pickle.dump(data, open(file_name, "wb"))
        else:
            raise ValueError("Invalid mode: {}".format(mode))
        return file_name

    def get_table_dict(self, ):
        return dict(self._tabular)

    def get_table_key_set(self, ):
        return set(key for key, value in self._tabular)

    @contextmanager
    def prefix(self, key):
        self.push_prefix(key)
        try:
            yield
        finally:
            self.pop_prefix()

    @contextmanager
    def tabular_prefix(self, key):
        self.push_tabular_prefix(key)
        yield
        self.pop_tabular_prefix()

    def log_variant(self, log_file, variant_data):
        mkdir_p(os.path.dirname(log_file))
        with open(log_file, "w") as f:
            json.dump(variant_data, f, indent=2, sort_keys=True, cls=MyEncoder)

    def record_tabular_misc_stat(self, key, values, placement='back'):
        if placement == 'front':
            prefix = ""
            suffix = key
        else:
            prefix = key
            suffix = ""
        if len(values) > 0:
            self.record_tabular(prefix + "Average" + suffix, np.average(values))
            self.record_tabular(prefix + "Std" + suffix, np.std(values))
            self.record_tabular(prefix + "Median" + suffix, np.median(values))
            self.record_tabular(prefix + "Min" + suffix, np.min(values))
            self.record_tabular(prefix + "Max" + suffix, np.max(values))
        else:
            self.record_tabular(prefix + "Average" + suffix, np.nan)
            self.record_tabular(prefix + "Std" + suffix, np.nan)
            self.record_tabular(prefix + "Median" + suffix, np.nan)
            self.record_tabular(prefix + "Min" + suffix, np.nan)
            self.record_tabular(prefix + "Max" + suffix, np.nan)

    def dump_tabular(self, *args, **kwargs):
        wh = kwargs.pop("write_header", None)
        if len(self._tabular) > 0:
            if self._log_tabular_only:
                self.table_printer.print_tabular(self._tabular)
            else:
                for line in tabulate(self._tabular).split('\n'):
                    self.log(line, *args, **kwargs)
            tabular_dict = dict(self._tabular)
            # Also write to the csv files
            # This assumes that the keys in each iteration won't change!
            for tabular_fd in list(self._tabular_fds.values()):
                writer = csv.DictWriter(tabular_fd,
                                        fieldnames=list(tabular_dict.keys()))
                if wh or (
                        wh is None and tabular_fd not in self._tabular_header_written):
                    writer.writeheader()
                    self._tabular_header_written.add(tabular_fd)
                writer.writerow(tabular_dict)
                tabular_fd.flush()
            del self._tabular[:]

    def pop_prefix(self, ):
        del self._prefixes[-1]
        self._prefix_str = ''.join(self._prefixes)

    def save_itr_params(self, itr, params):
        if self._snapshot_dir:
            if self._snapshot_mode == 'all':
                file_name = osp.join(self._snapshot_dir, 'itr_%d.pkl' % itr)
                pickle.dump(params, open(file_name, "wb"))
            elif self._snapshot_mode == 'last':
                # override previous params
                file_name = osp.join(self._snapshot_dir, 'params.pkl')
                pickle.dump(params, open(file_name, "wb"))
            elif self._snapshot_mode == "gap":
                if itr % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir, 'itr_%d.pkl' % itr)
                    pickle.dump(params, open(file_name, "wb"))
            elif self._snapshot_mode == "gap_and_last":
                if itr % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir, 'itr_%d.pkl' % itr)
                    pickle.dump(params, open(file_name, "wb"))
                file_name = osp.join(self._snapshot_dir, 'params.pkl')
                pickle.dump(params, open(file_name, "wb"))
            elif self._snapshot_mode == 'none':
                pass
            else:
                raise NotImplementedError

    def save_stats_pkl(self, epoch, stats):
        file_name = osp.join(self._snapshot_dir, 'stats_%d.pkl' % epoch)
        pickle.dump(stats, open(file_name, "wb"))

    def save_stats_csv(self, epoch, stats, fname='stats', numbered=True):
        base_name = '%s_%d.csv' % (fname, epoch) if numbered else '%s.csv' % fname
        file_name = osp.join(self._snapshot_dir, base_name)
        if not osp.isfile(file_name):
            with open(file_name, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=list(sorted(stats.keys())))
                writer.writeheader()
        with open(file_name, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=list(sorted(stats.keys())))
            writer.writerow(stats)

    def save_stats(self, epoch, stats, final=False):
        """ Final determines whether to process with `compute_stats` or not """
        if not final:
            save_stats = compute_stats(stats, horizon=0, task_obj=get_task_obj(osp.join(self._snapshot_dir, 'variant.json')))
        else:
            save_stats = stats  # save stats as-is
        if not osp.isfile(osp.join(self._snapshot_dir, 'validation_stats.csv')):
            with open(osp.join(self._snapshot_dir, 'validation_stats.csv'), 'a') as f:
                writer = csv.DictWriter(f, fieldnames=list(sorted(save_stats.keys())))
                writer.writeheader()
        with open(osp.join(self._snapshot_dir, 'validation_stats.csv'), 'a') as f:
            writer = csv.DictWriter(f, fieldnames=list(sorted(save_stats.keys())))
            writer.writerow(save_stats)

    def save_viz(self, epoch, params, array):
        np.save(join(self._snapshot_dir, 'visit_%d.npy' % epoch), array)

    def save_viz_old(self, epoch, params):
        def trim(arr):
            return arr[1:-1, 1:-1]

        def get_obs(env):
            img = env.get_img(onehot=env.one_hot_obs)
            full_img = env.get_full_img(scale=1 if env.fully_observed else 1 / 8, onehot=env.one_hot_obs)

            if env.fully_observed:
                obs = np.concatenate((full_img.flatten(), np.array(env.agent_pos)))
            elif env.only_partial_obs:
                obs = img.flatten()
            else:
                obs = np.concatenate((img.flatten(), full_img.flatten()))
            return obs

        qfs = params['trainer/qf']
        qfs.eval()
        env = params['exploration/env']
        visit_count = env.visit_count.copy()

        # loop over the grid to produce each possible state, and collect q-values
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # q function
        qs = np.zeros((len(qfs), env.grid_size, env.grid_size, len(env.actions)))
        for i in range(1, env.grid_size - 1):
            for j in range(1, env.grid_size - 1):
                if env.grid.get(i, j) is None:
                    env.agent_pos = np.array([i, j])
                    obs = get_obs(env)
                    obs = torch.from_numpy(obs).detach().float().to(device)
                    for idx, qf in enumerate(qfs):
                        q = qf(obs)
                        # swap i and j due to backwards minigrid coords
                        qs[idx, j, i] = q.cpu().detach()
        # value function
        vs = np.amax(qs, axis=3)

        variance_map = np.var(vs, axis=0)
        exp_q = np.exp(qs)
        # sum over ensemble
        sum_q = exp_q.sum(0)
        log_q = np.log(sum_q)
        # max over actions
        lse_map = np.amax(log_q, axis=2)

        # trim border
        visit_count = trim(visit_count)
        variance_map = trim(variance_map)
        lse_map = trim(lse_map)

        # # make plot
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 15))
        # ax1.set_title('LogSumExp value function')
        # im1 = ax1.imshow(lse_map)
        # ax2.set_title('Variance of ensemble')
        # im2 = ax2.imshow(variance_map)
        # ax3.set_title('Visitation counts')
        # im3 = ax3.imshow(visit_count)
        # fig.savefig(osp.join(self._snapshot_dir, 'map_itr_%d.png' % epoch))

def get_repo_dir():
    return osp.dirname(osp.dirname(osp.dirname(__file__)))

logger = Logger()
