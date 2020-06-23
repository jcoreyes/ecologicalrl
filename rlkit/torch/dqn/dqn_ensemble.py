from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class DQNEnsembleTrainer(TorchTrainer):
    def __init__(
            self,
            qfs,
            target_qfs,
            learning_rate=1e-3,
            soft_target_tau=1e-3,
            target_update_period=1,
            qf_criterion=None,
            grad_clip_val=None,
            discount=0.99,
            reward_scale=1.0,
    ):
        super().__init__()
        self.ensemble_size = len(qfs)
        self.qfs = qfs
        self.target_qfs = target_qfs
        self.learning_rate = learning_rate
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.qf_optimizer = optim.Adam(
            self.qfs.parameters(),
            lr=self.learning_rate,
            eps=1e-4,
        )
        self.discount = discount
        self.reward_scale = reward_scale
        self.qf_criterion = qf_criterion or nn.MSELoss()
        self.grad_clip_val = grad_clip_val
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        rewards = batch['rewards'] * self.reward_scale
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Compute loss
        """
        qf_losses = []
        for ensemble_idx in range(self.ensemble_size):
            qf = self.qfs[ensemble_idx]
            target_qf = self.target_qfs[ensemble_idx]

            target_q_values = target_qf(next_obs).detach().max(
                1, keepdim=True
            )[0]
            y_target = rewards + (1. - terminals) * self.discount * target_q_values
            y_target = y_target.detach()
            # actions is a one-hot vector
            y_pred = torch.sum(qf(obs) * actions, dim=1, keepdim=True)
            qf_loss = self.qf_criterion(y_pred, y_target)
            qf_losses.append(qf_loss)

            """
            Save some statistics for eval using just one batch.
            """
            if self._need_to_update_eval_statistics:
                if ensemble_idx == self.ensemble_size - 1:
                    self._need_to_update_eval_statistics = False
                self.eval_statistics['QF %d  Loss' % ensemble_idx] = np.mean(ptu.get_numpy(qf_loss))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Y %d Predictions' % ensemble_idx,
                    ptu.get_numpy(y_pred),
                ))

        """
        Soft target network updates
        """
        self.qf_optimizer.zero_grad()
        total_qf_loss = sum(qf_losses)
        total_qf_loss.backward()
        self.qf_optimizer.step()

        for ensemble_idx in range(self.ensemble_size):
            qf = self.qfs[ensemble_idx]
            target_qf = self.target_qfs[ensemble_idx]
            """
            Soft Updates
            """
            if self._n_train_steps_total % self.target_update_period == 0:
                ptu.soft_update_from_to(
                    qf, target_qf, self.soft_target_tau
                )

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.qfs,
            self.target_qfs,
        ]

    def get_snapshot(self):
        return dict(
            qf=self.qfs,
            target_qf=self.target_qfs,
        )
