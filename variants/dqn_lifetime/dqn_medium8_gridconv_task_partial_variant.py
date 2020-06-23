import math

from rlkit.policies.network_food import FoodNetworkEasy, FoodNetworkMediumFullObs, FoodNetworkMediumPartialObsTask
from rlkit.pythonplusplus import identity
from rlkit.torch.conv_networks import CNN
from rlkit.torch.networks import FlattenMlp, Mlp
from torch.nn import functional as F

variant = dict(
    # env_name="MiniGrid-Food-16x16-Medium-1Inv-2Tier-OneTime-Partial-Fixed-Lifespan200-v1",
    env_name="MiniGrid-Food-8x8-Medium-1Inv-2Tier-OneTime-Partial-Fixed-Lifespan500-v1",
    algorithm="DQN Lifetime",
    lifetime=True,
    version="normal",
    layer_size=256,
    replay_buffer_size=int(1E5),
    algorithm_kwargs=dict(
        # below two params don't matter
        num_epochs=3000,
        num_eval_steps_per_epoch=0,

        num_trains_per_train_loop=100,
        num_expl_steps_per_train_loop=50,
        min_num_steps_before_training=100,
        max_path_length=math.inf,
        batch_size=256,
    ),
    trainer_kwargs=dict(
        discount=0.99,
        learning_rate=3E-4,
    ),
    inventory_network_kwargs=dict(
        # shelf: 8 (repeated x8)
        input_size=64,
        output_size=16,
        hidden_sizes=[32],
    ),
    img_conv_kwargs=dict(
        # 5 grid
        input_width=5,
        input_height=5,
        # 16 channels
        input_channels=16,
        output_size=32,
        kernel_sizes=[3, 3],
        n_channels=[8, 8],
        strides=[1, 1],
        paddings=[1, 1],
        hidden_sizes=[64],
        batch_norm_conv=True
    )
)


def gen_network(variant, action_dim, layer_size, policy=False):
    return FoodNetworkMediumPartialObsTask(
        img_network=CNN(**variant['img_conv_kwargs']),
        inventory_network=FlattenMlp(**variant['inventory_network_kwargs']),
        final_network=FlattenMlp(
            input_size=variant['img_conv_kwargs']['output_size']
                       + variant['inventory_network_kwargs']['output_size'],
            output_size=action_dim,
            hidden_sizes=[layer_size, layer_size],
            output_activation=F.softmax if policy else identity
        ),
        sizes=[
            variant['img_conv_kwargs']['input_width'] * variant['img_conv_kwargs']['input_height'] *
            variant['img_conv_kwargs']['input_channels'],
            # shelf dim
            64
        ]
    )
