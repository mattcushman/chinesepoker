from CPServerSrc import CPGame

class CPMLGameEnv(object):
    def __init__(self, numPlayers, histLen):
        self.numPlayers = numPlayers
        self.histLen = histLen
    def reset(self, initgame=True):
        if initgame:
            self.game = CPGame.CPGame(list(range(self.numPlayers)))
        else:
            self.game=False
        self.playHist = []
        return self.actionHistory(self.histLen)
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
    def step(self, actionNum,game=False):
        return self.stepFull(self.getPossibleActions(game)[actionNum])
    def stepFull(self, action, game=False):
        if not game:
            game=self.game
        player=game.toMove
        game.implementMove([c for c in range(52) if action[c]==1])
        self.playHist.insert(0, action)
        return self.actionHistory(self.histLen), float(game.winner == player), self.done(), None
    def prettyState(self, game=False):
        if not game:
            game=self.game
        return game.prettyState()
    def actionHistory(self, n, game=False):
        if not game:
            game=self.game
            playHist=self.playHist
        else:
            playHist=[self.cardsToVector(c) for (p,c) in reversed(game.playerMoves)]
        if self.histLen > 0:
            return [self.cardsToVector(game.hands[game.toMove])] + playHist[:n] + [self.cardsToVector([]) for k in range(n-len(playHist))]
        else:
            return [self.cardsToVector(game.hands[game.toMove])] + playHist[:n]


