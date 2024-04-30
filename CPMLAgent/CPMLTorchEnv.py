from CPServerSrc import CPGame

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data import (
    BoundedTensorSpec, 
    CompositeSpec, 
    UnboundedContinuousTensorSpec,
    BinaryDiscreteTensorSpec,
    DiscreteTensorSpec
)
from torchrl.envs import CatTensors, EnvBase, Transform, TransformedEnv, UnsqueezeTransform


class CPMLTorchEnv(EnvBase):
    batch_locked = False
    def __init__(self, num_players=2, hist_len=64, seed=None, device="cpu", batch_size=[]):
        if len(batch_size) > 1:
            raise ValueError("CPMLTorchEnv does not support multidim batch_size > 1")
        super().__init__(device=device, batch_size=batch_size)
        self._make_spec(num_players, hist_len)
        self.num_players = num_players
        self.hist_len = hist_len
        self.action_history = torch.zeros(self.total_batch_size(), hist_len, 52, dtype=torch.bool)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    def _reset(self, tensordict):
        self.games = [CPGame.CPGame(list(range(self.num_players)),
                      deck=torch.randperm(52, generator=self.rng).tolist()) for _ in range(self.total_batch_size())]
        if len(self.batch_size) == 0:
            self.action_history = torch.zeros(self.hist_len, 52, dtype=torch.bool)
            game = self.games[0]
            return TensorDict({
                "reward": torch.tensor(0.0, dtype=torch.float32),
                "done": torch.tensor(False, dtype=torch.bool),
                "hand": torch.tensor([c in game.hands[game.toMove] for c in range(52)], dtype=torch.bool),
                "tomove": torch.tensor(game.toMove, dtype=torch.int64),
                "player": torch.tensor(game.toMove, dtype=torch.int64),
                "actionhistory": self.action_history,
            }, batch_size=self.batch_size,)
        else:
            self.action_history = torch.zeros(self.total_batch_size(), self.hist_len, 52, dtype=torch.float32)
            return TensorDict({
                "reward": torch.tensor([0.0 for game in self.games], dtype=torch.float32),
                "done": torch.tensor([False for game in self.games], dtype=torch.bool),
                "hand": torch.tensor([[c in game.hands[game.toMove] for c in range(52)]
                                    for game in self.games], dtype=torch.bool),
                "tomove": torch.tensor([game.toMove for game in self.games], dtype=torch.int64),
                "player": torch.tensor([game.toMove for game in self.games], dtype=torch.int64),
                "actionhistory": self.action_history,
            }, batch_size=self.batch_size,)
        
    def _step(self, tensordict):
        action = tensordict["action"]
        print("Action: ", action)
        reward = torch.zeros(self.batch_size, dtype=torch.float32)
        done = torch.zeros(self.batch_size, dtype=torch.bool)
        hand = torch.zeros(self.batch_size + ( 52, ), dtype=torch.bool)
        tomove = torch.zeros(self.batch_size, dtype=torch.int64)
        player = torch.zeros(self.batch_size, dtype=torch.int64)
        for i in range(self.total_batch_size()):
            game = self.games[i]
            this_player=game.toMove
            if len(self.batch_size) == 0:
                this_action = action
            else:
                this_action = action[i]
            try:
                game.implementMove([c for c in range(52) if this_action[c]])
            except CPGame.MoveError as move_error:
                print(f"Move error: {move_error.move} {move_error.msg}")
            if len(self.batch_size) == 0:
                torch.roll(self.action_history, 1, 0)
                self.action_history[0,:] = this_action
            else:
                torch.roll(self.action_history, 1, 1)
                self.action_history[i,0,:] = this_action
            if len(self.batch_size) == 0:
                reward = torch.tensor(float(game.winner == this_player))
                done = torch.tensor(game.done())
                hand = torch.tensor([c in game.hands[this_player] for c in range(52)], dtype=torch.bool)
                tomove = torch.tensor(game.toMove)
                player = torch.tensor(this_player)
            else:
                reward[i] = float(game.winner == this_player)
                done[i] = game.done()
                hand[i,:] = torch.tensor([c in game.hands[this_player] for c in range(52)], dtype=torch.bool)
                tomove[i] = game.toMove
                player[i] = this_player
        out = TensorDict({
            "reward": reward,
            "done": done,
            "hand": hand,
            "tomove": tomove,
            "player": player,
            "actionhistory": self.action_history,
        }, batch_size=self.batch_size,)
        return out
    
    def _make_spec(self, num_players, hist_len):
        self.observation_spec = CompositeSpec({
            "hand": BinaryDiscreteTensorSpec(52, dtype=torch.bool),
            "tomove": DiscreteTensorSpec(num_players),
            "player": DiscreteTensorSpec(num_players),
            "actionhistory": BinaryDiscreteTensorSpec(52, shape=(hist_len, 52), dtype=torch.bool)
        })
        self.action_spec = BinaryDiscreteTensorSpec(52, dtype=torch.bool)
        self.reward_spec = BoundedTensorSpec(low=0.0, high=1.0, shape=(1,), dtype=torch.float32)

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
        moves = []
        for i in range(self.total_batch_size()):
            game = self.games[i]
            game_moves = game.getMoves()
            game_move = game_moves[np.random.randint(len(game_moves))]
            moves.append([bool(i in game_move) for i in range(52)])
        if len(self.batch_size) == 0:
            r = TensorDict({"action": torch.tensor(moves[0], dtype=torch.bool)}, 
                               batch_size=self.batch_size)
        else:
            r = TensorDict({"action": torch.tensor(moves, dtype=torch.bool)}, batch_size=self.batch_size)
        if tensordict is None:
            return r
        tensordict.update(r)
        return tensordict
            

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
