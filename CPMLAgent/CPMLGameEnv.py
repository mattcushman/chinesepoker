from CPServerSrc import CPGame

class CPMLGameEnv(object):
    def __init__(self, numPlayers):
        self.numPlayers = numPlayers
    def reset(self):
        self.game = CPGame.CPGame(list(range(self.numPlayers)))
        self.playHist = []
    def cardsToVector(self, cards):
        return [int(x in cards) for x in range(52)]
    def getPossibleActions(self):
        moves=self.game.getMoves()
        return [self.cardsToVector(h) for h in moves]
    def done(self):
        return self.game.done()
    def step(self, action):
        player=self.game.toMove
        self.game.implementMove([c for c in range(52) if action[c]==1])
        self.playHist.insert(0, action)
        return self.playHist, int(self.game.winner == player), self.done(), None
    def prettyState(self):
        return self.game.prettyState()

