from rlkit.policies.network_food import FoodNetworkEasy, FoodNetworkMediumFullObs, FoodNetworkMediumPartialObsTask
from rlkit.pythonplusplus import identity
from rlkit.torch.conv_networks import CNN
from rlkit.torch.networks import FlattenMlp, Mlp
from torch.nn import functional as F

variant = dict(
    env_name="MiniGrid-Food-8x8-Medium-1Inv-2Tier-OneTime-Partial-v1",
    # env_name="MiniGrid-Food-8x8-Medium-1Inv-2Tier-Dense-Partial-Fixed-v1",
    # env_name="MiniGrid-Food-8x8-Medium-1Inv-2Tier-Dense-Partial-Random-v1",
    # env_name="MiniGrid-Food-8x8-Medium-1Inv-2Tier-OneTime-Partial-Lifespan400-v1",
    algorithm="DQN",
    version="normal",
    layer_size=128,
    replay_buffer_size=int(5E5),
    algorithm_kwargs=dict(
        num_epochs=1500,
        num_eval_steps_per_epoch=5000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=100,
        batch_size=512,
    ),
    trainer_kwargs=dict(
        discount=0.99,
        learning_rate=1E-4,
        grad_clip_val=5
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
        input_channels=8,
        output_size=128,
        kernel_sizes=[3, 3],
        n_channels=[32, 32],
        strides=[1, 1],
        paddings=[1, 1],
        hidden_sizes=[128, 128],
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
