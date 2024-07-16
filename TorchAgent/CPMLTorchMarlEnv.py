from CPServerSrc import CPGame

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn
import torch.nn.functional as fun

from torchrl.data import (
    BoundedTensorSpec, 
    CompositeSpec, 
    BinaryDiscreteTensorSpec,
    DiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    UnboundedContinuousTensorSpec
)
from torchrl.envs import CatTensors, EnvBase, Transform, TransformedEnv, UnsqueezeTransform
from torchrl.envs.utils import check_marl_grouping

def pad_list(l, length):
    if len(l) >= length:
        raise ValueError("Available moves is longer than length")
    return l + [[] for _ in range(length - len(l))]

class CPMLTorchMarlEnv(EnvBase):
    batch_locked = False
    AVAILABLE_ACTIONS_LEN = 128

    def __init__(self, num_envs=10, num_players=2, hist_len=64, seed=None, device="cpu"):
        self.num_players = num_players
        self.hist_len = hist_len
        self.num_envs = num_envs
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)
        self.agent_names = [f"player_{i}" for i in range(self.num_players)]
        self.group_map = {x:[x] for x in self.agent_names}
        self.categorical_actions = True # turns on group env?
        self.categorical_rewards = True
        super().__init__(device=device, batch_size=torch.Size((num_envs,)))
        self._make_spec(num_players, hist_len)

    def _reset(self, tensordict):
        if tensordict is None:
            self.games = [CPGame.CPGame(self.agent_names,
                        deck=torch.randperm(52, generator=self.rng).tolist()) 
                        for _ in range(self.num_envs)]
        else:
            for reset_index in range(self.num_envs):
                if tensordict["_reset"][reset_index]:
                    self.games[reset_index] = CPGame.CPGame(self.agent_names,
                                                            deck=torch.randperm(52, generator=self.rng).tolist()
                                                            ) 

        return TensorDict({
            agent_name: TensorDict({
               "hand": torch.tensor([[[c in game.hands[agent_name] for c in range(52)]] for game in self.games], 
                                    dtype=torch.int64, device=self.device),
                "available_actions": torch.tensor(
                    [
                        [[
                            [
                                c in move 
                                for c in range(52)
                            ] 
                            for move in pad_list(game.getMoves(agent_name), self.AVAILABLE_ACTIONS_LEN)
                        ]] 
                        for game in self.games
                    ], dtype=torch.int64, device=self.device
                ),
                "num_actions": torch.tensor(
                    [[[max([1,len(game.getMoves(agent_name))])]] for game in self.games], dtype=torch.int64, device=self.device
                ),
            }, batch_size=[self.num_envs,1])
            for agent_name in self.agent_names
        } | {
           "tomove": torch.tensor([[game.to_move_index()] for game in self.games], dtype=torch.int64, device=self.device),
           "actionhistory": torch.zeros(torch.Size([self.num_envs, self.hist_len, 52]), dtype=torch.int64, device=self.device),
           "done": torch.tensor([False] * self.num_envs, dtype=torch.bool, device=self.device)
        }, batch_size=[self.num_envs])
        
    def _step(self, tensordict):
        done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        actionhistory = torch.roll(tensordict["actionhistory"], 1, 1)
        for i in range(self.total_batch_size()):
            game = self.games[i]
            this_player=self.agent_names[tensordict["tomove"][i]]
            num_moves = int(tensordict[this_player]["num_actions"][i, 0, 0])
            action_index = torch.argmax(tensordict[this_player]["action"][i][0][0:num_moves])
            deck_action = tensordict[this_player]["available_actions"][i, 0, action_index]
            game.implementMove([c for c in range(52) if deck_action[c]])
            actionhistory[i, 0] = deck_action
            if len(self.batch_size) == 0:
                done = torch.tensor(game.done())
            else:
                done[i] = game.done()
        out = TensorDict({
            agent_name: TensorDict({
                "hand": torch.tensor([[[c in game.hands[agent_name] for c in range(52)]] for game in self.games], 
                                     dtype=torch.int64, device=self.device),
                "available_actions": torch.tensor(
                    [
                        [[
                            [
                                c in move 
                                for c in range(52)
                            ] 
                            for move in pad_list(game.getMoves(agent_name), self.AVAILABLE_ACTIONS_LEN)
                        ]] 
                        for game in self.games
                    ], dtype=torch.int64, device=self.device
                ),
                "num_actions": torch.tensor(
                    [[[max([1,len(game.getMoves(agent_name))])]] for game in self.games], dtype=torch.int64, device=self.device
                ),

                "reward": torch.tensor([[game.reward(agent_name)] for game in self.games], dtype=torch.float, device=self.device)
            }, batch_size=[self.num_envs,1])
            for agent_name in self.agent_names
        } | {
            "tomove": torch.tensor([[game.to_move_index()] for game in self.games], dtype=torch.int64, device=self.device),
            "actionhistory": actionhistory,
            "done": done
        }, batch_size=[self.num_envs])
        return out
    
    def _make_spec(self, num_players, hist_len):
        self.unbatched_action_spec = CompositeSpec(device=self.device)
        self.unbatched_observation_spec = CompositeSpec(device=self.device)
        self.unbatched_reward_spec = CompositeSpec(device=self.device)

        self.unbatched_observation_spec["tomove"] = DiscreteTensorSpec(n=num_players, shape=torch.Size((1,)), 
                                                                       device=self.device)
        self.unbatched_observation_spec["actionhistory"] = BinaryDiscreteTensorSpec(52, shape=(self.hist_len, 52), 
                                                                                    dtype=torch.int64,
                                                                                    device=self.device)

        for agent_name in self.agent_names:
            (
                group_observation_spec,
                group_action_spec,
                group_reward_spec,
            ) = self._make_unbatched_group_spec(agent_name)
            self.unbatched_observation_spec[agent_name] = group_observation_spec
            self.unbatched_action_spec[agent_name] = group_action_spec
            self.unbatched_reward_spec[agent_name] = group_reward_spec

        self.unbatched_done_spec = CompositeSpec(
            {
                "done": DiscreteTensorSpec(n=2, shape=torch.Size((1,)),
                                           dtype=torch.bool,
                                           device=self.device)
            }
        )
        
        self.observation_spec = self.unbatched_observation_spec.expand(*self.batch_size, 
                                                                       *self.unbatched_observation_spec.shape)
        self.action_spec = self.unbatched_action_spec.expand(*self.batch_size,
                                                            *self.unbatched_action_spec.shape)
        self.reward_spec = self.unbatched_reward_spec.expand(*self.batch_size,
                                                            *self.unbatched_reward_spec.shape)
        self.done_spec = self.unbatched_done_spec.expand(*self.batch_size, 
                                                         *self.unbatched_done_spec.shape)
        check_marl_grouping(self.group_map, self.agent_names)

    def _make_unbatched_group_spec(self, agent_name):
        action_spec = CompositeSpec({
            "action": OneHotDiscreteTensorSpec(self.AVAILABLE_ACTIONS_LEN, dtype=torch.int64)
        })
        observation_spec = CompositeSpec({
            "hand": BinaryDiscreteTensorSpec(52, dtype=torch.int64),
            "available_actions": BinaryDiscreteTensorSpec(52, shape=(self.AVAILABLE_ACTIONS_LEN, 52), dtype=torch.int64),
            "num_actions": DiscreteTensorSpec(self.AVAILABLE_ACTIONS_LEN, shape=(1,), dtype=torch.int64),
        })

        # It's important to wrap this in a CompositeSpec, as the agent_name is used as a key in the spec with "reward"
        reward_spec = CompositeSpec({
            "reward": UnboundedContinuousTensorSpec(shape=(1,), dtype=torch.float32)
        })

        group_observation_spec = torch.stack([observation_spec], dim=0)
        group_action_spec = torch.stack([action_spec], dim=0)
        group_reward_spec = torch.stack([reward_spec], dim=0)
        
        return group_observation_spec, group_action_spec, group_reward_spec


    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def encode_move(self, move):
        return [c in move for c in range(52)]

    def rand_action(self, tensordict: TensorDictBase | None = None) -> TensorDictBase:
        if tensordict is not None:
            shape = tensordict.shape
        elif not self.batch_size:
            shape = torch.Size([])
        elif tensordict.shape != self.batch_size:
            # if tensordict is not None and the env has a batch size, their shape must match
            raise RuntimeError(
                "The input tensordict and the env have a different batch size: "
                f"env.batch_size={self.batch_size} and tensordict.batch_size={tensordict.shape}. "
                f"Non batch-locked environment require the env batch-size to be either empty or to"
                f" match the tensordict one."
            )
        for agent_name in self.agent_names:
            num_actions_cpu = tensordict[agent_name]["num_actions"].cpu()
            random_actions = [[np.random.randint(num_actions_cpu[k][0][0])] for k in range(self.num_envs)]
            tensordict[agent_name]["action"] = fun.one_hot(
                torch.tensor(random_actions, device=self.device),
                num_classes=self.AVAILABLE_ACTIONS_LEN
            ) 
        return tensordict
            
    def total_batch_size(self):
        return self.num_envs

    def listStep(self, cardsList):
        self.vectorStep([int(x in cardsList) for x in range(52)])

    def vectorStep(self, cardsVector):
        cardsTensor = torch.tensor(cardsVector, dtype=torch.bool)
        return self.step(TensorDict({
            "action": cardsTensor
        }, batch_size=self.batch_size))

    def cardsToVector(self, cards):
        return [int(x in cards) for x in range(52)]
