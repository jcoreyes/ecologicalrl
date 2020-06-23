import torch

from rlkit.policies.base import Policy
from rlkit.torch.core import eval_np, torch_ify
from torch import nn
from torch.distributions import Categorical


class ImageNetwork(Policy, nn.Module):
    def __init__(self, img_network, final_network, sizes):
        """
        :param sizes: list of lengths of inputs in order present in observations
        """
        super().__init__()

        self.img_network = img_network
        self.final_network = final_network
        self.action_dim = final_network.output_size
        # length of inputs in order received
        self.sizes = sizes

    def forward(self, obs):
        # import pdb; pdb.set_trace()
        if len(obs.shape) < 2:
            obs = torch_ify(obs).unsqueeze(0)
        cumsum = 0
        arrs = []
        # import pdb;
        # pdb.set_trace()
        # for size in self.sizes:
        #     arrs.append(obs.narrow(dim=1, start=cumsum, length=size))
        #     cumsum += size

        # assert cumsum == obs.shape[1], 'not all of obs used'
        #
        # import pdb; pdb.set_trace()
        x = self.img_network(obs.contiguous().view((obs.shape[0], -1)))
        #import pdb; pdb.set_trace()

        out = self.final_network(x)
        return out

    def get_action(self, obs_np):
        dist_vec = eval_np(self, obs_np)
        action_idx = Categorical(torch_ify(dist_vec)).sample().item()
        return action_idx, {}


