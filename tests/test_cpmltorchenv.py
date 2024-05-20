# pytest module to test TorchAgent/CPMLTorchEnv.py
import pytest
import torch
from torchrl.envs.utils import check_env_specs
from torchrl.envs import TransformedEnv, RewardSum, MarlGroupMapType
import torch.nn.functional as fun
from tensordict import TensorDict

from TorchAgent.CPMLTorchEnv import CPMLTorchEnv
from TorchAgent.CPMLTorchMarlEnv import CPMLTorchMarlEnv

def test_CPMLTorchEnv():
    torch_env = CPMLTorchEnv(seed=1)
    assert torch_env.num_players == 2
    assert torch_env.hist_len == 64
    assert torch_env.action_history.shape == torch.Size([torch_env.total_batch_size(), 64, 52])

def test_check_env():
    torch_env = CPMLTorchEnv(seed=1)
    torch_env.reset()
    check_env_specs(torch_env)
                              
def test_CPMLTorchEnv_reset():
    torch_env = CPMLTorchEnv(seed=1)
    obs = torch_env.reset()
    assert torch_env.games is not None
    assert torch_env.games[0].toMove == 0

def test_CPMLTorchEnv_step():
    torch_env = CPMLTorchEnv(seed=2)
    obs = torch_env.reset()
    assert torch_env.games[0].toMove == 1
    assert len(torch_env.games[0].hands[1]) == 13
    assert len(torch_env.games[0].getMoves()) == 3
    torch_env.step(TensorDict({"action": fun.one_hot(torch.tensor(0), CPMLTorchEnv.AVAILABLE_ACTIONS_LEN)},  
                             batch_size=[])) 
    assert torch_env.games[0].toMove == 0
    assert len(torch_env.games[0].getMoves()) == 14
    torch_env.step(TensorDict({"action": fun.one_hot(torch.tensor(2), CPMLTorchEnv.AVAILABLE_ACTIONS_LEN)},  
                             batch_size=[])) 
    assert torch_env.games[0].toMove == 1
    assert len(torch_env.games[0].getMoves()) == 12
    
def test_CMPLtorch_env_rollout():
    torch_env = CPMLTorchEnv(seed=3)
    obs = torch_env.reset()
    assert torch_env.games[0].toMove == 1
    obs = torch_env.rollout(5)
    assert torch_env.games[0].toMove == 0
    action_history_1 = torch_env.action_history.numpy()
    obs = torch_env.rollout(5)
    assert torch_env.games[0].toMove == 1
    action_history_2 = torch_env.action_history.numpy()
    for i in range(5):
        assert (action_history_1[0,i] == action_history_2[0,5+i]).all()
    obs = torch_env.rollout(10)
    action_history_3 = torch_env.action_history.numpy()
    for i in range(10):
        assert (action_history_2[0,i] == action_history_3[0,10+i]).all()

def test_CMPLtorch_env_fullgame():
    torch_env = CPMLTorchEnv(seed=4)
    obs = torch_env.reset()
    while not torch_env.games[0].done():
        print("*"*80)
        print(torch_env.games[0].prettyState())
        actions_list = torch_env.games[0].getMoves()
        action_lengths = [len(move) for move in actions_list]
        action_index = action_lengths.index(max(action_lengths))
        for i, move in enumerate(actions_list):
            print(f"{i}: {torch_env.games[0].cardsToString(move)}")
        print(f"action index = {action_index}")
        print(f"action = {actions_list[action_index]}")
        obs = torch_env.step(TensorDict({"action": fun.one_hot(torch.tensor(action_index), 
                                                              CPMLTorchEnv.AVAILABLE_ACTIONS_LEN)},  
                             batch_size=[]))

def test_CMPLtorch_env_fullgame_hardcode():
    move_to_make = [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 8, 1, 1, 1, 0, 5, 0, 0, 0, 0, 0, 0]
    torch_env = CPMLTorchEnv(seed=4)
    obs = torch_env.reset()
    for move_number,move_index in enumerate(move_to_make):
        assert not torch_env.games[0].done()
        obs = torch_env.step(TensorDict({"action": fun.one_hot(torch.tensor(move_index), 
                                                              CPMLTorchEnv.AVAILABLE_ACTIONS_LEN)},  
                             batch_size=[]))
        if move_number < len(move_to_make)-1:
            assert obs["next"]["done"] == False
            assert obs["next"]["reward"] == torch.tensor(0.0)
    assert torch_env.games[0].done()
    assert torch_env.games[0].winner == 0
    assert torch_env.games[0].toMove == 1
    assert obs["next"]["done"] == True
    assert obs["next"]["reward"] == torch.tensor(+1.0)

def test_check_env_specs_marl():
    torch_env = CPMLTorchMarlEnv(seed=1)
    torch_env.reset()
    check_env_specs(torch_env)