"""
Torch ensemble policy
"""
import numpy as np
from torch import nn

import rlkit.torch.pytorch_util as ptu
from rlkit.policies.base import Policy


class EnsembleArgmaxDiscretePolicy(nn.Module, Policy):
    def __init__(self, qfs):
        super().__init__()
        self.qfs = qfs

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = ptu.from_numpy(obs).float()

        all_q_values = []
        for qf in self.qfs:
            q_values = qf(obs).squeeze(0)
            q_values_np = ptu.get_numpy(q_values)
            all_q_values.append(q_values_np)
        all_q_values = np.stack(all_q_values, 0)
        actions = all_q_values.argmax(1)
        majority_vote = np.bincount(actions).argmax()  # Choose action with highest occurence

        return majority_vote, {}


class EnsembleLSEDiscretePolicy(nn.Module, Policy):
    def __init__(self, qfs):
        super().__init__()
        self.qfs = qfs

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = ptu.from_numpy(obs).float()

        all_q_values = []
        for qf in self.qfs:
            q_values = qf(obs).squeeze(0)
            q_values_np = ptu.get_numpy(q_values)
            all_q_values.append(q_values_np)
        all_q_values = np.stack(all_q_values, 0)
        exp_q = np.exp(all_q_values)
        sum_q = exp_q.sum(0)
        log_q = np.log(sum_q)
        action = log_q.argmax()
        return action, {}
