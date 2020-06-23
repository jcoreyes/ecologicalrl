import math

from rlkit.policies.network_food import FoodNetworkEasy, FoodNetworkMediumFullObs, \
    FoodNetworkMediumPartialObsTask, FoodNetworkPartialObsGoal
from rlkit.pythonplusplus import identity
from rlkit.torch.conv_networks import CNN
from rlkit.torch.networks import FlattenMlp, Mlp
from torch.nn import functional as F

variant = dict(
    env_name="MiniGrid-Food-8x8-Medium-1Inv-GoalLifetime-Random-v1",
    algorithm="DQN Lifetime",
    version="normal",
    lifetime=True,
    layer_size=16,
    replay_buffer_size=int(5E5),
    algorithm_kwargs=dict(
        num_epochs=1500,
        num_eval_steps_per_epoch=5000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=math.inf,
        batch_size=512,
    ),
    trainer_kwargs=dict(
        discount=0.99,
        learning_rate=1E-4,
        grad_clip_val=5
    ),
    inventory_network_kwargs=dict(
        # shelf: 8 x 8
        input_size=64,
        output_size=16,
        hidden_sizes=[16, 16],
    ),
    img_network_kwargs=dict(
        # 5 x 5 x 8
        input_size=200,
        output_size=32,
        hidden_sizes=[64, 64]
    ),
    goal_network_kwargs=dict(
        # 8 x 8
        input_size=64,
        output_size=16,
        hidden_sizes=[16, 16]
    )
)


def gen_network(variant, action_dim, layer_size, policy=False):
    return FoodNetworkPartialObsGoal(
        img_network=Mlp(**variant['img_network_kwargs']),
        inventory_network=FlattenMlp(**variant['inventory_network_kwargs']),
        goal_network=FlattenMlp(**variant['goal_network_kwargs']),
        final_network=FlattenMlp(
            input_size=variant['img_network_kwargs']['output_size']
                       + variant['inventory_network_kwargs']['output_size']
                       + variant['goal_network_kwargs']['output_size'],
            output_size=action_dim,
            hidden_sizes=[layer_size, layer_size],
            output_activation=F.softmax if policy else identity
        ),
        sizes=[
            variant['img_network_kwargs']['input_size'],
            # shelf dim
            64,
            # goal dim
            64
        ]
    )
