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
    OneHotDiscreteTensorSpec
)
from torchrl.envs import CatTensors, EnvBase, Transform, TransformedEnv, UnsqueezeTransform


def pad_list(l, length):
    if len(l) >= length:
        raise ValueError("Available moves is longer than length")
    return l + [[] for _ in range(length - len(l))]

class CPMLTorchEnv(EnvBase):
    batch_locked = False
    AVAILABLE_ACTIONS_LEN = 64

    def __init__(self, num_players=2, hist_len=64, seed=None, device="cpu", batch_size=[]):
        if len(batch_size) > 1:
            raise ValueError("CPMLTorchEnv does not support multidim batch_size > 1")
        super().__init__(device=device, batch_size=batch_size)
        self._make_spec(num_players, hist_len)
        self.num_players = num_players
        self.hist_len = hist_len
        self.action_history = torch.zeros(self.total_batch_size(), hist_len, 52, dtype=torch.int64)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _reset(self, tensordict):
        self.games = [CPGame.CPGame(list(range(self.num_players)),
                      deck=torch.randperm(52, generator=self.rng).tolist()) for _ in range(self.total_batch_size())]
        if len(self.batch_size) == 0:
            self.action_history = torch.zeros(self.hist_len, 52, dtype=torch.int64)
            game = self.games[0]
            self.available_actions = torch.tensor([[[c in move for c in range(52)] 
                                                   for move in pad_list(game.getMoves(), 
                                                                        self.AVAILABLE_ACTIONS_LEN)]], 
                                                   dtype=torch.int64)
            return TensorDict({
                "done": torch.tensor(False, dtype=torch.bool),
                "hand": torch.tensor([c in game.hands[game.toMove] for c in range(52)], dtype=torch.int64),
                "tomove": torch.tensor(game.toMove, dtype=torch.int64),
                "available_actions": self.available_actions_value(),
                "actionhistory": self.action_history,
            }, batch_size=self.batch_size,)
        else:
            self.action_history = torch.zeros(self.total_batch_size(), self.hist_len, 52, dtype=torch.float32)
            self.available_actions = torch.tensor([[[c in move for c in range(52)] 
                                                    for move in pad_list(game.getMoves(), 
                                                                         self.AVAILABLE_ACTIONS_LEN)]
                                                    for game in self.games], 
                                                    dtype=torch.int64)
            return TensorDict({
                "done": torch.tensor([False for game in self.games], dtype=torch.bool),
                "hand": torch.tensor([[c in game.hands[game.toMove] for c in range(52)]
                                    for game in self.games], dtype=torch.int64),
                "tomove": torch.tensor([game.toMove for game in self.games], dtype=torch.int64),
                "available_actions": self.available_actions_value(),
                "actionhistory": self.action_history,
            }, batch_size=self.batch_size,)
        
    def _step(self, tensordict):
        action = tensordict["action"]
        if len(self.batch_size) == 0:
            action_index = torch.argmax(action)
        else:
            action_index = torch.argmax(action, dim=1)
        reward = torch.zeros(self.batch_size, dtype=torch.float)
        done = torch.zeros(self.batch_size, dtype=torch.bool)
        hand = torch.zeros(self.batch_size + ( 52, ), dtype=torch.int64)
        tomove = torch.zeros(self.batch_size, dtype=torch.int64)
        for i in range(self.total_batch_size()):
            game = self.games[i]
            this_player=game.toMove
            if len(self.batch_size) == 0:
                this_action_index = action_index
            else:
                this_action_index = action_index[i]
            deck_action = self.available_actions[i][this_action_index]
            try:
                game.implementMove([c for c in range(52) if deck_action[c]])
            except CPGame.MoveError as move_error:
                print(f"Move error: {move_error.move} {move_error.msg}")
            if len(self.batch_size) == 0:
                self.action_history = torch.roll(self.action_history, 1, 0)
                self.action_history[0,:] = deck_action
            else:
                self.action_history = torch.roll(self.action_history, 1, 1)
                self.action_history[i,0,:] = deck_action
            if len(self.batch_size) == 0:
                reward = torch.tensor(float(game.winner == this_player))
                done = torch.tensor(game.done())
                hand = torch.tensor([c in game.hands[this_player] for c in range(52)], dtype=torch.int64)
                tomove = torch.tensor(game.toMove)
            else:
                reward[i] = float(game.winner == this_player)
                done[i] = game.done()
                hand[i,:] = torch.tensor([c in game.hands[this_player] for c in range(52)], dtype=torch.int64)
                tomove[i] = game.toMove
            self.available_actions[i] = torch.tensor([[c in move for c in range(52)] 
                                                for move in pad_list(game.getMoves(), self.AVAILABLE_ACTIONS_LEN)], 
                                                dtype=torch.int64)
        out = TensorDict({
            "reward": reward,
            "done": done,
            "hand": hand,
            "tomove": tomove,
            "actionhistory": self.action_history,
            "available_actions": self.available_actions_value(),
        }, batch_size=self.batch_size,)
        return out
    
    def _make_spec(self, num_players, hist_len):
        self.observation_spec = CompositeSpec({
            "hand": BinaryDiscreteTensorSpec(52, dtype=torch.int64),
            "tomove": DiscreteTensorSpec(num_players),
            "available_actions": BinaryDiscreteTensorSpec(52, shape=(self.AVAILABLE_ACTIONS_LEN, 52), dtype=torch.int64),
            "actionhistory": BinaryDiscreteTensorSpec(52, shape=(hist_len, 52), dtype=torch.int64)
        })
        self.action_spec = CompositeSpec({
            "action": OneHotDiscreteTensorSpec(self.AVAILABLE_ACTIONS_LEN, dtype=torch.int64)
        })
        self.reward_spec = CompositeSpec({
            "reward": BoundedTensorSpec(low=0.0, high=1.0, shape=(1,), dtype=torch.float32)
        })

    def _set_seed(self, seed):
        rng = torch.manual_seed(seed)
        self.rng = rng

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
        moves = [np.random.randint(len(game.getMoves())) for game in self.games]
        if len(self.batch_size) == 0:
            r = TensorDict({"action": fun.one_hot(torch.tensor(moves[0]), num_classes=self.AVAILABLE_ACTIONS_LEN)}, 
                               batch_size=self.batch_size)
        else:
            r = TensorDict({"action": fun.one_hot(torch.tensor(moves), num_classes=self.AVAILABLE_ACTIONS_LEN)}, 
                           batch_size=self.batch_size)
        if tensordict is None:
            return r
        tensordict.update(r)
        return tensordict
            
    def available_actions_value(self):
        if len(self.batch_size) == 0:
            return self.available_actions[0]
        else:
            return self.available_actions

    def total_batch_size(self):
        total_batch_size = 1
        for i in self.batch_size:
            total_batch_size *= i
        return total_batch_size

    def listStep(self, cardsList):
        self.vectorStep([int(x in cardsList) for x in range(52)])

    def vectorStep(self, cardsVector):
        cardsTensor = torch.tensor(cardsVector, dtype=torch.bool)
        return self.step(TensorDict({
            "action": cardsTensor
        }, batch_size=self.batch_size))

    def cardsToVector(self, cards):
        return [int(x in cards) for x in range(52)]
    def getPossibleActions(self, game=False):
        if not game:
            game=self.game
        moves=game.getMoves()
        return [self.cardsToVector(h) for h in moves]
    def done(self, game=False):
        if not game:
            game=self.game
        return game.done()
    def prettyState(self, game=False):
        if not game:
            game=self.game
        return game.prettyState()
