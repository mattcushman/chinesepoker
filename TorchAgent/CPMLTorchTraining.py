import numpy as np
import argparse
import copy

import torch
from torch import nn
from torch import optim
from torchrl.modules import MLP, MultiAgentMLP, Actor, EGreedyModule
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import TransformedEnv, RewardSum, DTypeCastTransform, Compose, check_env_specs
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, RandomSampler, ReplayBuffer
from torchrl.objectives import DDPGLoss, SoftUpdate, ValueEstimators
from torchrl.envs.utils import step_mdp

from TorchAgent.CPMLTorchMarlEnv import CPMLTorchMarlEnv

import multiprocessing

from tqdm import tqdm

from CPServerSrc.CPGame import cardsToString

def group_observation_keys(group):
    return ["tomove", "actionhistory", (group, "hand"), (group, "available_actions"), (group, "num_actions")]

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
            in_keys=group_observation_keys(group),
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
                    num_actions,
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
        policy_net = MLP(
            in_features=num_observation_dimensions(env, group),
            out_features=env.AVAILABLE_ACTIONS_LEN,
            depth=4,
            num_cells=256,
            activation_class=nn.ReLU,
            device=env.device,
        )
        policy_modules[group] = policy_net
    return policy_modules

def make_policies(env, policy_modules):
    policies = {}
    for group, policy_module in policy_modules.items():
        group_observation_key = (group, "all_observations")

        cat_module = make_observation_module(env, group, group_observation_key)

        actor = Actor(
            module = policy_module,
            spec = env.full_action_spec[(group, "action")],
            in_keys = group_observation_key,
            out_keys = [(group, "action")],
        )

        policies[group] = TensorDictSequential(cat_module, actor)
    return policies

def make_critics(args, env):
    critics = {}
    for group in env.agent_names:
        obs_action_cat_module = make_observation_action_module(env, group, (group, "obs_action_cat"))

        critic_module = TensorDictModule(
            module=MultiAgentMLP(
                n_agent_inputs=num_observation_dimensions(env, group) + env.AVAILABLE_ACTIONS_LEN,
                n_agent_outputs=1,
                n_agents=1,
                share_params=args.share_params_critic,
                device=env.device,
                depth=3,
                num_cells=256,
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
                args.memory_size,
            ),
            sampler=RandomSampler(),
            batch_size=args.training_batch_size,
        )
    return replay_buffers

def create_loss_functions(env, policy_modules, critics, args):
    losses = {}
    target_updaters = {}
    for group in env.agent_names:
        loss_module = DDPGLoss(
            actor_network = policy_modules[group],
            value_network = critics[group],
            delay_value = True,
            loss_function = "l2",
#            reduction="mean",
        )
        loss_module.set_keys(
            state_action_value=(group, "state_action_value"),
            reward=(group, "reward"),
            done=(group, "done"),
            terminated=(group, "terminated"),
        )
        loss_module.make_value_estimator(ValueEstimators.TD0, gamma=args.gamma)        
        # loss_module.make_value_estimator(ValueEstimators.TDLambda)        
        losses[group] = loss_module

    target_updaters = {group: SoftUpdate(loss, tau=args.polyak_tau) for group, loss in losses.items()}

    return losses, target_updaters

def create_optimizers(losses, args):
    return {
        group: {
            "loss_actor": torch.optim.Adam(
                loss.actor_network_params.flatten_keys().values(), lr=args.learning_rate
            ),
            "loss_value": torch.optim.Adam(
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

def pretty_print_game(actionhistory):
    out = ""
    for i in range(actionhistory.shape[0]):
        cards = [card for card in range(52) if actionhistory[i,card] == 1]
        out += f"Move {i}: {cardsToString(cards)}\n"
    return out

def save_sample_trajectories(policies, critics, name, n=10):
    # open file named "name" for writing
    # create 10 trajectories and output the game as logged in the environment game
    with open(name, "w") as f:
        for i in range(n):
            f.write("*******************************************************\n")
            f.write(f"Game number {i}\n")
            env = create_environment(1001+i, 1, torch.device("cpu"))
            obs = env.reset()
            while not env.games[0].done():
                player = env.games[0].toMove
                f.write(env.games[0].prettyState()+"\n")
                device = policies[player].device
                action = policies[player](obs.to(device))
                obs = env.step(action)
                value = critics[player](obs)[(player, "state_action_value")].item()
                f.write(f"Critic value: {value:8.4f}\n")
                obs = step_mdp(obs, keep_other=True)
                cards = [card for card in range(52) if obs["actionhistory"][0,0,card] == 1]
                f.write(f"Player {player} move: {cardsToString(cards)}\n")
                f.write("   ---")


def save_models(args, policies, critics):
    for group, policy in policies.items():
        torch.save(policy.state_dict(), f"{args.save_model_path}/policy_{group}_policy.pth")
    for group, critic in critics.items():
        torch.save(critic.state_dict(), f"{args.save_model_path}/critic_{group}_critic.pth")

def create_environment(seed, num_envs, device):
    base_env = CPMLTorchMarlEnv(seed=seed,
                               num_envs=num_envs,
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
    return env

# define the main function
def main(args=None):
    # parse command line parameters
    args = parse_args(args)

    device = setup()
    # create the environment

    env = create_environment(args.seed, args.num_envs, device)

    policy_modules = make_policy_modules(env)
    policies = make_policies(env, policy_modules)
    critics = make_critics(args, env)

    # do we need to create some exploration policies here?
    exploration_modules = {}
    exploration_policies = {}
    for group, policy in policies.items():
        exploration_modules[group] = EGreedyModule(
            spec=env.full_action_spec[(group,"action")], 
            eps_init=0.25, 
            eps_end=0.02, 
            annealing_num_steps=1000,
            action_key=(group, "action"),
        )

        exploration_policies[group] = TensorDictSequential(
            policy,
            exploration_modules[group]
        )

    # Data collection
    agents_exploration_policy = TensorDictSequential(*exploration_policies.values())

    collector = SyncDataCollector(
        env, 
        agents_exploration_policy,
        device=device,
        frames_per_batch=args.frames_per_batch,
        total_frames=args.num_episodes * args.frames_per_batch,
    )

    replay_buffers = create_replay_buffers(args, device, env)

    losses, target_updaters = create_loss_functions(env, policies, critics, args)

    optimizers = create_optimizers(losses, args)

    progress_bar = tqdm(
        total = args.num_episodes,
        desc  = " ".join([f"espiode_reward_mean_{group}" for group in env.agent_names]),
    )

    episode_reward_mean_map = {group: [] for group in env.agent_names}

    for iteration, batch in enumerate(collector):
        current_frames = batch.numel()
        batch = process_batch(batch, env)
        for group in env.agent_names:
            group_batch = batch.exclude(
                *[key for _group in env.agent_names if _group != group for key in [_group, ("next", _group)]]
            )
            group_batch = group_batch.reshape(-1)
            replay_buffers[group].extend(group_batch)

            for _ in range(args.num_optimizer_steps):
                subdata = replay_buffers[group].sample().to(device)
                loss_vals = losses[group](subdata)

                for loss_name in ["loss_actor", "loss_value"]:
                    loss = loss_vals[loss_name]
                    optimizer = optimizers[group][loss_name]
                    loss.backward()

                    # is this clipping correct?
                    params = optimizer.param_groups[0]["params"]
                    torch.nn.utils.clip_grad_norm_(params, 1.0)

                    optimizer.step()
                    optimizer.zero_grad()
            
                target_updaters[group].step()

            exploration_modules[group].step(current_frames)

        for group in env.agent_names:
            episode_reward_mean = batch.get(("next", group, "reward"))[batch.get(("next", group, "done"))].mean().item()
            episode_reward_mean_map[group].append(episode_reward_mean)

        progress_bar.set_description(
            ", ".join([f"espiode_reward_mean_{group}: {episode_reward_mean_map[group][-1]:.2f}" for group in env.agent_names]),
            refresh=False,
        )
        progress_bar.update()

        save_sample_trajectories(policies, critics, name=f"{args.sample_trajectories_dir}/sample_trajectories_{iteration}", n=10)

    save_models(args, policies, critics)


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
    parser.add_argument("--num-episodes", type=int, default=1100)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--frames_per_batch", type=int, default=1024)
    parser.add_argument("--training-batch-size", type=int, default=512)
    parser.add_argument("--num-optimizer-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.00001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--polyak-tau", type=float, default=0.5)
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--save-model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--share-params-critic", type=bool, default=True)
    parser.add_argument("--centralised-critic", type=bool, default=True)
    parser.add_argument("--memory-size", type=int, default=1000000)
    parser.add_argument("--save_model_path", type=str, default="saved_models")
    parser.add_argument("--sample_trajectories_dir", type=str, default="sample_trajectories")
    return parser.parse_args(args)

if __name__ == "__main__":
    main()
    assert True
    print("Done.")
