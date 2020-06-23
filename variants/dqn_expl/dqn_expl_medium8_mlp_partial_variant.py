from rlkit.policies.network_food import FoodNetworkEasy, FoodNetworkMediumFullObs, FoodNetworkMediumPartialObs
from rlkit.pythonplusplus import identity
from rlkit.torch.conv_networks import CNN
from rlkit.torch.networks import FlattenMlp, Mlp
from torch.nn import functional as F

variant = dict(
    env_name="MiniGrid-Food-8x8-Medium-1Inv-2Tier-Dense-Partial-Random-v1",
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
        # pantry: 400, shelf: 8, health:1
        input_size=409,
        output_size=64,
        hidden_sizes=[128, 128],
    ),
    img_network_kwargs=dict(
        # 7 x 7 x 2
        input_size=98,
        output_size=32,
        hidden_sizes=[64, 64]
    )
)


def gen_network(variant, action_dim, layer_size, policy=False):
    return FoodNetworkMediumPartialObs(
        img_network=Mlp(**variant['img_network_kwargs']),
        inventory_network=FlattenMlp(**variant['inventory_network_kwargs']),
        final_network=FlattenMlp(
            input_size=variant['img_network_kwargs']['output_size']
                       + variant['inventory_network_kwargs']['output_size'],
            output_size=action_dim,
            hidden_sizes=[layer_size, layer_size],
            output_activation=F.softmax if policy else identity
        ),
        sizes=[
            variant['img_network_kwargs']['input_size'],
            # health dim
            1,
            # pantry dim
            400,
            # shelf dim
            8
        ]
    )
