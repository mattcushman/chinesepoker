# pytest module to test CPMLAgent/CPMLTorchEnv.py
import pytest
import torch
from torchrl.envs.utils import check_env_specs

from CPMLAgent.CPMLTorchEnv import CPMLTorchEnv

def test_CPMLTorchEnv():
    torchenv = CPMLTorchEnv(seed=1)
    assert torchenv.num_players == 2
    assert torchenv.hist_len == 64
    assert torchenv.action_history.shape == torch.Size([torchenv.total_batch_size(), 64, 52])

def test_check_env():
    torchenv = CPMLTorchEnv(seed=1)
    torchenv.reset()
    check_env_specs(torchenv)
                              
def test_CPMLTorchEnv_reset():
    torchenv = CPMLTorchEnv(seed=1)
    obs = torchenv.reset()
    assert torchenv.games is not None
    assert torchenv.games[0].toMove == 0

def test_CPMLTorchEnv_step():
    torchenv = CPMLTorchEnv(seed=2)
    obs = torchenv.reset()
    assert torchenv.games[0].toMove == 1
    assert len(torchenv.games[0].hands[1]) == 13
    assert len(torchenv.games[0].getMoves()) == 3
    torchenv.listStep(torchenv.games[0].getMoves()[0]) 
    assert torchenv.games[0].toMove == 0
    assert len(torchenv.games[0].getMoves()) == 14
    torchenv.listStep(torchenv.games[0].getMoves()[2])
    assert torchenv.games[0].toMove == 1
    assert len(torchenv.games[0].getMoves()) == 12
    