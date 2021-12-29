import numpy as np

from CPServerSrc import CPGame

class CPMLGameEnv(object):
    def __init__(self, numPlayers, histLen):
        self.numPlayers = numPlayers
        self.histLen = histLen
    def reset(self):
        self.game = CPGame.CPGame(list(range(self.numPlayers)))
        self.playHist = []
        return self.actionHistory(self.histLen)
    def cardsToVector(self, cards):
        return [int(x in cards) for x in range(52)]
    def getPossibleActions(self):
        moves=self.game.getMoves()
        return [self.cardsToVector(h) for h in moves]
    def done(self):
        return self.game.done()
    def step(self, actionNum):
        return self.stepFull(self.getPossibleActions()[actionNum])
    def stepFull(self, action):
        player=self.game.toMove
        self.game.implementMove([c for c in range(52) if action[c]==1])
        self.playHist.insert(0, action)
        return self.actionHistory(self.histLen), float(self.game.winner == player), self.done(), None
    def prettyState(self):
        return self.game.prettyState()
    def actionHistory(self, n):
        return [self.cardsToVector(self.game.hands[self.game.toMove])] + self.playHist[:n] + [self.cardsToVector([]) for k in range(n-len(self.playHist))]


