import numpy as np
import argparse
import torch
from torch import nn
from torch import optim
from torchrl.modules import MultiAgentMLP, Actor, EGreedyModule
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import TransformedEnv, RewardSum, DTypeCastTransform, Compose, check_env_specs
from torchrl.collectors import SyncDataCollector
from TorchAgent.CPMLTorchMarlEnv import CPMLTorchMarlEnv
import multiprocessing

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

# def make_lookup_module(group):
#     return TensorDictModule(
#         lambda action, available_actions: torch.gather(
#             available_actions, 
#             2, 
#             torch.argmax(action,dim=-1)[..., None, None].long()
#         ),
#         in_keys=[(group, "action_index"), (group, "available_actions")],
#         out_keys=[(group, "action")],
#     )

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
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--update-target-every", type=int, default=100)
    parser.add_argument("--save-model-every", type=int, default=100)
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--save-model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--share-params-critic", type=bool, default=True)
    parser.add_argument("--centralised-critic", type=bool, default=True)
    return parser.parse_args(args)

if __name__ == "__main__":
    main()
