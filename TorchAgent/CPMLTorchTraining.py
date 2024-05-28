import numpy as np
import argparse
import torch
from torch import nn
from torch import optim
from torchrl.modules import MultiAgentMLP
from tensordict.nn import TensorDictModule
from TorchAgent.CPMLTorchMarlEnv import CPMLTorchMarlEnv
import multiprocessing

def make_policy_modules(env):
    policy_modules = {}
    for group in env.agent_names:
        num_inputs = (env.observation_spec["tomove"].shape[-1] +
                      env.observation_spec["actionhistory"].shape[-1] * env.observation_spec["actionhistory"].shape[-2] +
                      env.observation_spec[group, "hand"].shape[-1] +
                      env.observation_spec[group, "available_actions"].shape[-1] * env.observation_spec[group, "available_actions"].shape[-2] +
                      env.observation_spec[group, "num_actions"].shape[-1]
        )
        policy_net = MultiAgentMLP(
            n_agent_inputs=num_inputs,
            n_agent_outputs=env.full_action_spec[group].shape[-1],
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
            in_keys=["tomove", "actionhistory", (group, "hand"), (group, "available_actions"), (group, "num_actions")],
            out_keys=[(group, "param")],
        )
        policy_modules[group] = policy_module
    return policy_modules



# define the main function
def main():
    # parse command line parameters
    args = parse_args()
    device = setup()
    # create the environment
    env = CPMLTorchMarlEnv(seed=args.seed, device=device)
    policy_modules = make_policy_modules(env)

def setup():
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    return device


# parse command line parameters for machine learning training
def parse_args():
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
    return parser.parse_args()

if __name__ == "__main__":
    main()