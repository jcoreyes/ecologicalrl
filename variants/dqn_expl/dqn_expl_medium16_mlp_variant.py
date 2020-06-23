from rlkit.policies.network_food import FoodNetworkEasy, FoodNetworkMediumFullObs
from rlkit.pythonplusplus import identity
from rlkit.torch.conv_networks import CNN
from rlkit.torch.networks import FlattenMlp, Mlp
from torch.nn import functional as F

variant = dict(
    env_name="MiniGrid-Food-16x16-Medium-1Inv-2Tier-Dense-v1",
    algorithm="DQN-Exploration",
    version="normal",
    layer_size=256,
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
    inventory_network_kwargs=dict(
        # pantry: 400, shelf: 8, health:1, pos:2
        input_size=411,
        output_size=64,
        hidden_sizes=[128, 128],
    ),
    full_img_network_kwargs=dict(
        # 16 x 16 x 2
        input_size=512,
        output_size=64,
        hidden_sizes=[128, 128]
    )
)


def gen_network(variant, action_dim, layer_size, policy=False):
    return FoodNetworkMediumFullObs(
        full_img_network=Mlp(**variant['full_img_network_kwargs']),
        inventory_network=FlattenMlp(**variant['inventory_network_kwargs']),
        final_network=FlattenMlp(
            input_size=variant['full_img_network_kwargs']['output_size']
                       + variant['inventory_network_kwargs']['output_size'],
            output_size=action_dim,
            hidden_sizes=[layer_size, layer_size],
            output_activation=F.softmax if policy else identity
        ),
        sizes=[
            variant['full_img_network_kwargs']['input_size'],
            # health dim
            1,
            # agent pos dim
            2,
            # pantry dim
            400,
            # shelf dim
            8
        ]
    )
