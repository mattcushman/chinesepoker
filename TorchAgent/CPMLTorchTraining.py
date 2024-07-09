import numpy as np
import argparse

import torch
from torch import nn
from torch import optim
from torchrl.modules import MultiAgentMLP, Actor, EGreedyModule
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import TransformedEnv, RewardSum, DTypeCastTransform, Compose, check_env_specs
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, RandomSampler, ReplayBuffer
from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators

from TorchAgent.CPMLTorchMarlEnv import CPMLTorchMarlEnv

import multiprocessing

from tqdm import tqdm

def make_observation_module(env, group, out_key="all_observations"):
    return TensorDictModule(
            lambda  tomove, actionhistory, hand, available_actions, num_actions: 
                torch.cat([
                    tomove[:,None,:], 
                    torch.flatten(actionhistory, start_dim=1)[:,None,:], 
                    hand, 
                    torch.flatten(available_actions, start_dim=2), 
                    num_actions
                ], dim=-1)
            ,
            in_keys=["tomove", "actionhistory", (group, "hand"), (group, "available_actions"), (group, "num_actions")],
            out_keys=[out_key],
        )

def make_observation_action_module(env, group, out_key="obs_action_cat"):
    return TensorDictModule(
            lambda  tomove, actionhistory, hand, available_actions, num_actions, action: 
                torch.cat([
                    tomove[:,None,:], 
                    torch.flatten(actionhistory, start_dim=1)[:,None,:], 
                    hand, 
                    torch.flatten(available_actions, start_dim=2), 
                    num_actions[:,None,:],
                    action
                ], dim=-1)
            ,
            in_keys=["tomove", "actionhistory", (group, "hand"), (group, "available_actions"), (group, "num_actions"), (group, "action")],
            out_keys=[out_key],
        )

def num_observation_dimensions(env, group):
    return (env.observation_spec["tomove"].shape[-1] +
            env.observation_spec["actionhistory"].shape[-1] * env.observation_spec["actionhistory"].shape[-2] +
            env.observation_spec[group, "hand"].shape[-1] +
            env.observation_spec[group, "available_actions"].shape[-1] * env.observation_spec[group, "available_actions"].shape[-2] +
            env.observation_spec[group, "num_actions"].shape[-1]
            )

def make_policy_modules(env):
    policy_modules = {}
    for group in env.agent_names:
        group_observation_key = (group, "all_observations")
        cat_module = make_observation_module(env, group, group_observation_key)
        policy_net = MultiAgentMLP(
            n_agent_inputs=num_observation_dimensions(env, group),
            n_agent_outputs=env.AVAILABLE_ACTIONS_LEN,
            n_agents=1,
            share_params=False,
            device=env.device,
            depth=2,
            num_cells=128,
            activation_class=nn.ReLU,
            centralised=False,
        )
        policy_module = TensorDictModule(
            policy_net,
            in_keys=[group_observation_key],
            out_keys=[(group, "action")],
        )
        # lookup_module = make_lookup_module(group)

        policy_modules[group] = TensorDictSequential(cat_module, policy_module)
    return policy_modules

def make_policies(env, policy_modules):
    policies = {}
    for group, policy_module in policy_modules.items():
        policies[group] = Actor(
            module = policy_module,
            spec = env.full_action_spec[(group, "action")],
            in_keys=[(group, "param")],
            out_keys=[(group, "action")],
            safe = True,
        )
    return policies

def make_critics(args, env):
    critics = {}
    for group in env.agent_names:
        obs_action_cat_module = make_observation_action_module(env, group, (group, "obs_action_cat"))

        critic_module = TensorDictModule(
            module=MultiAgentMLP(
                n_agent_inputs=num_observation_dimensions(env, group) + env.full_action_spec[group].shape[-1],
                n_agent_outputs=1,
                n_agents=1,
                share_params=args.share_params_critic,
                device=env.device,
                depth=2,
                num_cells=128,
                activation_class=nn.ReLU,
                centralised=args.centralised_critic,
            ),
            in_keys=[(group, "obs_action_cat")],
            out_keys=[(group, "state_action_value")],
        )

        critics[group] = TensorDictSequential(obs_action_cat_module, critic_module)
    return critics

def create_replay_buffers(args, device, env):
    replay_buffers = {}
    for group in env.agent_names:
        replay_buffers[group] = ReplayBuffer(
            storage=LazyMemmapStorage(
                args.memory_size,  #device=device
            ),
            sampler=RandomSampler(),
            batch_size=args.training_batch_size,
        )
    return replay_buffers

def create_loss_functions(env, policy_modules, critics, replay_buffers, args):
    losses = {}
    target_updaters = {}
    for group in env.agent_names:
        loss_module = DDPGLoss(
            actor_network = policy_modules[group],
            value_network = critics[group],
            delay_value = True,
            loss_function = "l2",
        )
        loss_module.set_keys(
            state_action_value=(group, "state_action_value"),
            reward=(group, "reward"),
            done=(group, "done"),
            terminated=(group, "terminated"),
        )
        loss_module.make_value_estimator(ValueEstimators.TD0, gamma=args.gamma)        
        losses[group] = loss_module
        target_updaters[group] = SoftUpdate(loss_module, tau=args.polyak_tau)
    return losses, target_updaters

def create_optimizers(losses, args):
    return {
        group: {
            "loss_actor": torch.optim.Adam(
                loss.actor_network_params.flatten_keys().values(), lr=args.learning_rate
            ),
            "loss_critic": torch.optim.Adam(
                loss.value_network_params.flatten_keys().values(), lr=args.learning_rate
            ),
        }
        for group, loss in losses.items()
    }

def process_batch(batch: TensorDictBase, env) -> TensorDictBase:
    """
    If the `(group, "terminated")` and `(group, "done")` keys are not present, create them by expanding
    `"terminated"` and `"done"`.
    This is needed to present them with the same shape as the reward to the loss.
    """
    for group in env.group_map.keys():
        keys = list(batch.keys(True, True))
        group_shape = batch.get_item_shape(group)
        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        if nested_done_key not in keys:
            batch.set(
                nested_done_key,
                batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if nested_terminated_key not in keys:
            batch.set(
                nested_terminated_key,
                batch.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
            )
    return batch


# define the main function
def main(args=None):
    # parse command line parameters
    args = parse_args(args)

    device = setup()
    # create the environment
    base_env = CPMLTorchMarlEnv(seed=args.seed,
                               num_envs=args.batch_size,
                               device=device
                               )
    
    long_to_float_transform = DTypeCastTransform(torch.int64, torch.float32,
                                                 in_keys=["actionhistory", 
                                                          ("player_0", "hand"), ("player_0", "available_actions"),
                                                          ("player_1", "hand"), ("player_1", "available_actions")
                                                 ],
                                                 in_keys_inv=[]
                                                 )

    env = TransformedEnv(
        base_env,
        long_to_float_transform,
    )

    policy_modules = make_policy_modules(env)
    # policies = make_policies(env, policy_modules)
    critics = make_critics(args, env)

    td = env.reset()
    print(td)
    td = policy_modules['player_0'](td)
    td = policy_modules['player_1'](td)
    print('************************************************************************************************')
    print(td)
    print('************************************************************************************************')
    env.step(td)

    # do we need to create some exploration policies here?
    exploration_policies = {}
    for group, policy in policy_modules.items():
        exploration_policies[group] = TensorDictSequential(
            policy,
            EGreedyModule(spec=env.full_action_spec[(group,"action")], 
                          eps_init=0.2, 
                          eps_end=0.01, 
                          annealing_num_steps=1000,
                          action_key=(group, "action"))
        )

    # Data collection
    agents_exploration_policy = TensorDictSequential(*exploration_policies.values())

    print(policy_modules['player_0'])



    env.rollout(policy=TensorDictSequential(*policy_modules.values()), max_steps=5)

    collector = SyncDataCollector(
        env, 
        agents_exploration_policy,
        device=device,
        frames_per_batch=args.batch_size,
        total_frames=args.num_episodes * args.batch_size,
    )

    replay_buffers = create_replay_buffers(args, device, env)

    losses, target_updaters = create_loss_functions(env, policy_modules, critics, replay_buffers, args)

    optimizers = create_optimizers(losses, args)

    progress_bar = tqdm(
        total = args.num_episodes,
        desc  = " ".join([f"espiode_reward_mean_{group}" for group in env.agent_names]),
    )

    episode_reward_mean_map = {group: [] for group in env.group_maps.keys()}
    train_group_map = copy.deepcopy(env.group_map)

    for episode, batch in enumerate(collector):
        current_frames = batch.numel()
        batch = process_batch(batch, env)
        for group in env.agent_names:
            group_batch = batch.exclude(
                *[key for _group in env.agent_names if _group != group for key in [_group, ("next", _group)]]
            )
            group_batch = group_batch.reshape(-1)
            replay_buffers[group].extend(group_batch)

            for _ in range(args.num_optimizer_steps):
                subdata = replay_buffers[group].sample()
                loss_vals = losses[group](subdata)

                for loss_name in ["loss_actor", "loss_critic"]:
                    loss = loss_vals[loss_name]
                    optimizer = optimizers[group][loss_name]
                    loss.backward()

                    # do i nbeed to add clipping here?
                    optimizer.step()
                    optimizer.zero_grad()
            
            target_updaters[group].step()

        exploration_policies[group].step(current_frames)

        if iteration == args.num_episodes//2:
            del train_group_map["agent"]

        for group in env.agent_names:
            episode_reward_mean = batch.get(("next", group, "reward"))[batch.get(("next", group, "done"))].mean().item()
            episode_reward_mean_map[group].append(episode_reward_mean)

        progress_bar.set_description(
            ", ".join([f"espiode_reward_mean_{group}: {episode_reward_mean_map[group][-1]:.2f}" for group in env.agent_names]),
            refresh=False,
        )
        progress_bar.update()


def setup():
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    return device


# parse command line parameters for machine learning training
def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-batch-size", type=int, default=128)
    parser.add_argument("--num-optimizer-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--polyak-tau", type=float, default=0.005)
    parser.add_argument("--update-target-every", type=int, default=100)
    parser.add_argument("--save-model-every", type=int, default=100)
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--save-model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--share-params-critic", type=bool, default=True)
    parser.add_argument("--centralised-critic", type=bool, default=True)
    parser.add_argument("--memory-size", type=int, default=1000000)
    return parser.parse_args(args)

if __name__ == "__main__":
    main()
    assert False
