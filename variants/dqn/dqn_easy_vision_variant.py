from rlkit.policies.network_food import FoodNetworkEasy
from rlkit.torch.conv_networks import CNN
from rlkit.torch.networks import FlattenMlp
from torch.nn import functional as F

variant = dict(
    env_name="MiniGrid-Food-8x8-Easy-6and4-Vision-v1",
    algorithm="SAC Discrete",
    version="normal",
    layer_size=64,
    replay_buffer_size=int(1E5),
    algorithm_kwargs=dict(
        num_epochs=3000,
        num_eval_steps_per_epoch=5000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=3000,
        batch_size=256,
    ),
    trainer_kwargs=dict(
        discount=0.99,
        learning_rate=3E-4,
    ),
    img_conv_kwargs=dict(
        # 7 grid * 8 pixel/grid
        input_width=56,
        input_height=56,
        # 3 rgb channels
        input_channels=3,
        output_size=64,
        kernel_sizes=[3, 3, 3],
        n_channels=[16, 16, 16],
        strides=[1, 1, 1],
        paddings=[1, 1, 1],
        hidden_sizes=[256, 128],
    ),
    full_img_conv_kwargs=dict(
        # 8 grid * 4 pixel/grid
        input_width=32,
        input_height=32,
        # 3 rgb channels
        input_channels=3,
        output_size=64,
        kernel_sizes=[3, 3, 3],
        n_channels=[16, 32, 32],
        strides=[1, 1, 1],
        paddings=[1, 1, 1],
        hidden_sizes=[512, 256],
    )
)


def gen_network(variant, action_dim, layer_size, policy=False):
    final_network_kwargs = dict(
        # +1 for health
        input_size=variant['img_conv_kwargs']['output_size'] + variant['full_img_conv_kwargs']['output_size'] + 1,
        output_size=action_dim,
        hidden_sizes=[layer_size, layer_size],
    )
    if policy:
        final_network_kwargs.update(output_activation=F.softmax)
    return FoodNetworkEasy(
        img_network=CNN(**variant['img_conv_kwargs']),
        full_img_network=CNN(**variant['full_img_conv_kwargs']),
        final_network=FlattenMlp(**final_network_kwargs),
        sizes=[
            variant['img_conv_kwargs']['input_width'] * variant['img_conv_kwargs']['input_height'] *
            variant['img_conv_kwargs']['input_channels'],
            variant['full_img_conv_kwargs']['input_width'] * variant['full_img_conv_kwargs']['input_height'] *
            variant['full_img_conv_kwargs']['input_channels'],
            # health dim
            1
        ]
    )
