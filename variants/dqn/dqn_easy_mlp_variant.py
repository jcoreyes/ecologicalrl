import math

from rlkit.policies.network_food import FoodNetworkEasy
from rlkit.pythonplusplus import identity
from rlkit.torch.conv_networks import CNN
from rlkit.torch.networks import FlattenMlp, Mlp
from rlkit.torch.sac.policies import CategoricalPolicy
from torch.nn import functional as F

variant = dict(
    env_name="MiniGrid-Food-8x8-Easy-6and4-v1",
    # env_name="MiniGrid-Food-8x8-Easy-10and6-Cap50-Decay-Lifespan30-v1",
    algorithm="DQN",
    version="normal",
    layer_size=32,
    replay_buffer_size=int(1E5),
    algorithm_kwargs=dict(
        num_epochs=3000,
        num_eval_steps_per_epoch=10000,
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
    network_kwargs=dict(
        input_size=227
    )
)


def gen_network(variant, action_dim, layer_size, policy=False):
    if policy:
        network = CategoricalPolicy(
            Mlp(
                input_size=variant['network_kwargs']['input_size'],
                output_size=action_dim,
                hidden_sizes=[layer_size, layer_size],
                output_activation=F.softmax
            )
        )
    else:
        network = FlattenMlp(
            input_size=variant['network_kwargs']['input_size'],
            output_size=action_dim,
            hidden_sizes=[layer_size, layer_size]
        )

    return network
