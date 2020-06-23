from collections import OrderedDict

import numpy as np
from rlkit.torch.core import np_to_pytorch_batch
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class HierarchyTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            setter,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            target_setter,

            low_qf,
            low_target_qf,
            policy,

            target_setter_noise=0.2,
            target_setter_noise_clip=0.5,
            setter_and_target_update_period=2,
            tau=0.005,

            discount=0.99,
            reward_scale=1.0,

            setter_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,
            grad_clip_val=None,

            # soft_target_tau=1e-2,
            # target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
    ):
        super().__init__()
        self.env = env
        self.setter = setter
        self.target_setter = target_setter
        self.target_setter_noise = target_setter_noise
        self.target_setter_noise_clip = target_setter_noise_clip

        # High
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        # Low
        self.low_qf = low_qf
        self.low_target_qf = low_target_qf
        self.policy = policy

        self._low_num_train_steps = 0
        self._high_num_train_steps = 0
        self.setter_and_target_update_period = setter_and_target_update_period
        self.tau = tau
        self.grad_clip_val = grad_clip_val

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=setter_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.setter_optimizer = optimizer_class(
            self.setter.parameters(),
            lr=setter_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.low_qf_optimizer = optimizer_class(
            self.low_qf.parameters(),
            lr=qf_lr
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        raise NotImplementedError

    def low_train(self, np_batch):
        self._low_num_train_steps += 1
        batch = np_to_pytorch_batch(np_batch)
        self.low_train_from_torch(batch)

    def high_train(self, np_batch):
        self._high_num_train_steps += 1
        batch = np_to_pytorch_batch(np_batch)
        self.high_train_from_torch(batch)

    def high_train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Critic operations.
        """

        next_actions = self.target_setter(next_obs)
        noise = ptu.randn(next_actions.shape) * self.target_setter_noise
        noise = torch.clamp(
            noise,
            -self.target_setter_noise_clip,
            self.target_setter_noise_clip
        )
        noisy_next_actions = next_actions + noise

        target_q1_values = self.target_qf1(next_obs, noisy_next_actions)
        target_q2_values = self.target_qf2(next_obs, noisy_next_actions)
        target_q_values = torch.min(target_q1_values, target_q2_values)
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()

        q1_pred = self.qf1(obs, actions)
        bellman_errors_1 = (q1_pred - q_target) ** 2
        qf1_loss = bellman_errors_1.mean()

        q2_pred = self.qf2(obs, actions)
        bellman_errors_2 = (q2_pred - q_target) ** 2
        qf2_loss = bellman_errors_2.mean()

        """
        Update Networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        setter_actions = setter_loss = None
        if self._n_train_steps_total % self.setter_and_target_update_period == 0:
            setter_actions = self.setter(obs)
            q_output = self.qf1(obs, setter_actions)
            setter_loss = - q_output.mean()

            self.setter_optimizer.zero_grad()
            setter_loss.backward()
            self.setter_optimizer.step()

            ptu.soft_update_from_to(self.setter, self.target_setter, self.tau)
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.tau)

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            if setter_loss is None:
                setter_actions = self.setter(obs)
                q_output = self.qf1(obs, setter_actions)
                setter_loss = - q_output.mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Setter Loss'] = np.mean(ptu.get_numpy(
                setter_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors 1',
                ptu.get_numpy(bellman_errors_1),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors 2',
                ptu.get_numpy(bellman_errors_2),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Setter Action',
                ptu.get_numpy(setter_actions),
            ))
        self._n_train_steps_total += 1

    def low_train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goals = batch['goals']
        # kinda an approximation since doesn't account for goal switching
        next_goals = self.setter.goal_transition(obs, goals, next_obs)

        """
        Compute loss
        """
        best_action_idxs = self.low_qf(torch.cat((next_obs, next_goals), dim=1)).max(1, keepdim=True)[1]
        target_q_values = self.low_target_qf(torch.cat((next_obs, next_goals), dim=1)).gather(
            1, best_action_idxs
        ).detach()
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        # actions is a one-hot vector
        y_pred = torch.sum(self.low_qf(torch.cat((obs, goals), dim=1)) * actions, dim=1, keepdim=True)
        qf_loss = self.qf_criterion(y_pred, y_target)

        """
        Update networks
        """
        self.low_qf_optimizer.zero_grad()
        qf_loss.backward()
        if self.grad_clip_val is not None:
            nn.utils.clip_grad_norm_(self.low_qf.parameters(), self.grad_clip_val)
        self.low_qf_optimizer.step()

        """
        Soft target network updates
        """
        if self._n_train_steps_total % self.setter_and_target_update_period == 0:
            ptu.soft_update_from_to(
                self.low_qf, self.low_target_qf, self.tau
            )

        """
        Save some statistics for eval using just one batch.
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Y Predictions',
                ptu.get_numpy(y_pred),
            ))

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_setter,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            target_policy=self.target_setter,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
        )
