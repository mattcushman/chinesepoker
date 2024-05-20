import numpy as np
import torch
from torch import nn
from torch import optim
from CPMLAgent  import CPMLGameEnv
from CPMLAgent.CPMLModelDef import create_q_model, get_action_probs, get_group_action_probs, num_players, hist_len, loss_function


def setup():
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    return device
)

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

