import torch

from rlkit.policies.base import Policy
from rlkit.torch.core import eval_np, torch_ify
from torch import nn
from torch.distributions import Categorical


class FoodNetworkMedium(Policy, nn.Module):
    def __init__(self, img_network, full_img_network, inventory_network, final_network, sizes):
        """
        :param sizes: list of lengths of inputs in order present in observations
        """
        super().__init__()

        self.img_network = img_network
        self.full_img_network = full_img_network
        self.inventory_network = inventory_network
        self.final_network = final_network
        self.action_dim = final_network.output_size
        # length of inputs in order received
        self.sizes = sizes

    def forward(self, obs):
        if len(obs.shape) < 2:
            obs = torch_ify(obs).unsqueeze(0)
        cumsum = 0
        arrs = []
        for size in self.sizes:
            arrs.append(obs.narrow(dim=1, start=cumsum, length=size))
            cumsum += size
        assert cumsum == obs.shape[1], 'not all of obs used'

        img, full_img, health, pantry, shelf = arrs
        x_img = self.img_network(img)
        x_full_img = self.full_img_network(full_img)
        x_inventory = self.inventory_network(pantry, shelf, health)
        out = self.final_network(torch.cat((x_img, x_full_img, x_inventory), dim=1))

        return out

    def get_action(self, obs_np):
        dist_vec = eval_np(self, obs_np)
        return Categorical(torch_ify(dist_vec)).sample().item(), {}


class FoodNetworkMediumFullObs(Policy, nn.Module):
    def __init__(self, full_img_network, inventory_network, final_network, sizes):
        """
        :param sizes: list of lengths of inputs in order present in observations
        """
        super().__init__()

        self.full_img_network = full_img_network
        self.inventory_network = inventory_network
        self.final_network = final_network
        self.action_dim = final_network.output_size
        # length of inputs in order received
        self.sizes = sizes

    def forward(self, obs):
        if len(obs.shape) < 2:
            obs = torch_ify(obs).unsqueeze(0)
        cumsum = 0
        arrs = []
        for size in self.sizes:
            arrs.append(obs.narrow(dim=1, start=cumsum, length=size))
            cumsum += size
        assert cumsum == obs.shape[1], 'not all of obs used'

        full_img, health, pos, pantry, shelf = arrs
        x_full_img = self.full_img_network(full_img)
        x_inventory = self.inventory_network(pantry, shelf, health, pos)
        out = self.final_network(torch.cat((x_full_img, x_inventory), dim=1))

        return out

    def get_action(self, obs_np):
        dist_vec = eval_np(self, obs_np)
        return Categorical(torch_ify(dist_vec)).sample().item(), {}


class FoodNetworkMediumFullObsTask(Policy, nn.Module):
    def __init__(self, img_network, inventory_network, final_network, sizes):
        """
        :param sizes: list of lengths of inputs in order present in observations
        """
        super().__init__()

        self.img_network = img_network
        self.inventory_network = inventory_network
        self.final_network = final_network
        self.action_dim = final_network.output_size
        # length of inputs in order received
        self.sizes = sizes

    def forward(self, obs):
        if len(obs.shape) < 2:
            obs = torch_ify(obs).unsqueeze(0)
        cumsum = 0
        arrs = []
        for size in self.sizes:
            arrs.append(obs.narrow(dim=1, start=cumsum, length=size))
            cumsum += size
        assert cumsum == obs.shape[1], 'not all of obs used (%d of %d)' % (cumsum, obs.shape[1])

        img, pos, shelf = arrs
        x_img = self.img_network(img)
        x_inventory = self.inventory_network(shelf, pos)
        out = self.final_network(torch.cat((x_img, x_inventory), dim=1))

        return out

    def get_action(self, obs_np):
        dist_vec = eval_np(self, obs_np)
        return Categorical(torch_ify(dist_vec)).sample().item(), {}


class FoodNetworkMediumPartialObsTask(Policy, nn.Module):
    def __init__(self, img_network, inventory_network, final_network, sizes):
        """
        :param sizes: list of lengths of inputs in order present in observations
        """
        super().__init__()

        self.img_network = img_network
        self.inventory_network = inventory_network
        self.final_network = final_network
        self.action_dim = final_network.output_size
        # length of inputs in order received
        self.sizes = sizes

    def forward(self, obs):
        if len(obs.shape) < 2:
            obs = torch_ify(obs).unsqueeze(0)
        cumsum = 0
        arrs = []
        for size in self.sizes:
            arrs.append(obs.narrow(dim=1, start=cumsum, length=size))
            cumsum += size
        #assert cumsum == obs.shape[1], 'not all of obs used'

        img, shelf = arrs
        x_img = self.img_network(img)
        x_inventory = self.inventory_network(shelf)
        out = self.final_network(x_img, x_inventory)
        # out = self.final_network(torch.cat((x_img, x_inventory), dim=1))

        return out

    def get_action(self, obs_np):
        dist_vec = eval_np(self, obs_np)
        return Categorical(torch_ify(dist_vec)).sample().item(), {}


class FoodNetworkMediumPartialObsTaskNumObj(Policy, nn.Module):
    def __init__(self, img_network, inventory_network, num_obj_network, final_network, sizes):
        """
        :param sizes: list of lengths of inputs in order present in observations
        """
        super().__init__()

        self.img_network = img_network
        self.inventory_network = inventory_network
        self.num_obj_network = num_obj_network
        self.final_network = final_network
        self.action_dim = final_network.output_size
        # length of inputs in order received
        self.sizes = sizes

    def forward(self, obs):
        if len(obs.shape) < 2:
            obs = torch_ify(obs).unsqueeze(0)
        cumsum = 0
        arrs = []
        for size in self.sizes:
            arrs.append(obs.narrow(dim=1, start=cumsum, length=size))
            cumsum += size
        assert cumsum == obs.shape[1], 'not all of obs used'

        img, shelf, num_obj = arrs
        x_img = self.img_network(img)
        x_inventory = self.inventory_network(shelf)
        x_num_obj = self.num_obj_network(num_obj)
        out = self.final_network(x_img, x_inventory, x_num_obj)

        return out

    def get_action(self, obs_np):
        dist_vec = eval_np(self, obs_np)
        return Categorical(torch_ify(dist_vec)).sample().item(), {}


class FoodNetworkMediumPartialObsTaskHealth(Policy, nn.Module):
    def __init__(self, img_network, inventory_network, final_network, sizes):
        """
        :param sizes: list of lengths of inputs in order present in observations
        """
        super().__init__()

        self.img_network = img_network
        self.inventory_network = inventory_network
        self.final_network = final_network
        self.action_dim = final_network.output_size
        # length of inputs in order received
        self.sizes = sizes

    def forward(self, obs):
        if len(obs.shape) < 2:
            obs = torch_ify(obs).unsqueeze(0)
        cumsum = 0
        arrs = []
        for size in self.sizes:
            arrs.append(obs.narrow(dim=1, start=cumsum, length=size))
            cumsum += size
        assert cumsum == obs.shape[1], 'not all of obs used'

        img, shelf, health = arrs
        x_img = self.img_network(img)
        x_inventory = self.inventory_network(shelf)
        out = self.final_network(x_img, x_inventory, health)

        return out

    def get_action(self, obs_np):
        dist_vec = eval_np(self, obs_np)
        return Categorical(torch_ify(dist_vec)).sample().item(), {}


class FoodNetworkPartialObsGoal(Policy, nn.Module):
    def __init__(self, img_network, inventory_network, goal_network, final_network, sizes):
        """
        :param sizes: list of lengths of inputs in order present in observations
        """
        super().__init__()

        self.img_network = img_network
        self.inventory_network = inventory_network
        self.goal_network = goal_network
        self.final_network = final_network
        self.action_dim = final_network.output_size
        # length of inputs in order received
        self.sizes = sizes

    def forward(self, obs):
        if len(obs.shape) < 2:
            obs = torch_ify(obs).unsqueeze(0)
        cumsum = 0
        arrs = []
        for size in self.sizes:
            arrs.append(obs.narrow(dim=1, start=cumsum, length=size))
            cumsum += size
        assert cumsum == obs.shape[1], 'not all of obs used'

        img, shelf, goal = arrs
        x_img = self.img_network(img)
        x_inventory = self.inventory_network(shelf)
        x_goal = self.goal_network(goal)
        out = self.final_network(x_img, x_inventory, x_goal)

        return out

    def get_action(self, obs_np):
        dist_vec = eval_np(self, obs_np)
        return Categorical(torch_ify(dist_vec)).sample().item(), {}


class FoodNetworkMediumPartialObs(Policy, nn.Module):
    def __init__(self, img_network, inventory_network, final_network, sizes):
        """
        :param sizes: list of lengths of inputs in order present in observations
        """
        super().__init__()

        self.img_network = img_network
        self.inventory_network = inventory_network
        self.final_network = final_network
        self.action_dim = final_network.output_size
        # length of inputs in order received
        self.sizes = sizes

    def forward(self, obs):
        if len(obs.shape) < 2:
            obs = torch_ify(obs).unsqueeze(0)
        cumsum = 0
        arrs = []
        for size in self.sizes:
            arrs.append(obs.narrow(dim=1, start=cumsum, length=size))
            cumsum += size
        assert cumsum == obs.shape[1], 'not all of obs used'

        img, health, pantry, shelf = arrs
        x_img = self.img_network(img)
        x_inventory = self.inventory_network(pantry, shelf, health)
        out = self.final_network(torch.cat((x_img, x_inventory), dim=1))

        return out

    def get_action(self, obs_np):
        dist_vec = eval_np(self, obs_np)
        return Categorical(torch_ify(dist_vec)).sample().item(), {}


class FlatFoodNetworkMedium(Policy, nn.Module):
    def __init__(self, img_network, full_img_network, inventory_network, final_network, sizes):
        """
        :param sizes: list of lengths of inputs in order present in observations
        """
        super().__init__()

        self.img_network = img_network
        self.full_img_network = full_img_network
        self.inventory_network = inventory_network
        self.final_network = final_network
        self.action_dim = final_network.output_size
        # length of inputs in order received
        self.sizes = sizes

    def forward(self, obs):
        if len(obs.shape) < 2:
            obs = torch_ify(obs).unsqueeze(0)
        cumsum = 0
        arrs = []
        for size in self.sizes:
            arrs.append(obs.narrow(dim=1, start=cumsum, length=size))
            cumsum += size
        assert cumsum == obs.shape[1], 'not all of obs used'

        img, full_img, health, pantry, shelf = arrs
        x_img = self.img_network(img)
        x_full_img = self.full_img_network(full_img)
        x_inventory = self.inventory_network(pantry, shelf, health)
        out = self.final_network(torch.cat((x_img, x_full_img, x_inventory), dim=1))

        return out

    def get_action(self, obs_np):
        dist_vec = eval_np(self, obs_np)
        return Categorical(torch_ify(dist_vec)).sample().item(), {}


class FoodNetworkEasy(Policy, nn.Module):
    def __init__(self, img_network, full_img_network, final_network, sizes):
        """
        :param sizes: list of lengths of inputs in order present in observations
        """
        super().__init__()

        self.img_network = img_network
        self.full_img_network = full_img_network
        self.final_network = final_network
        self.action_dim = final_network.output_size
        # length of inputs in order received
        self.sizes = sizes

    def forward(self, obs):
        if len(obs.shape) < 2:
            obs = torch_ify(obs).unsqueeze(0)
        cumsum = 0
        arrs = []
        for size in self.sizes:
            arrs.append(obs.narrow(dim=1, start=cumsum, length=size))
            cumsum += size
        assert cumsum == obs.shape[1], 'not all of obs used'

        img, full_img, health = arrs
        x_img = self.img_network(img)
        x_full_img = self.full_img_network(full_img)
        out = self.final_network(torch.cat((x_img, x_full_img, health), dim=1))

        return out

    def get_action(self, obs_np):
        dist_vec = eval_np(self, obs_np)
        return Categorical(torch_ify(dist_vec)).sample().item(), {}
