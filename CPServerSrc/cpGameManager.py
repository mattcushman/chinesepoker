from CPGame import CPGame
from CPMLAgent import CPMLModelDef
from CPMLAgent import CPMLGameEnv
from tensorflow import keras

class NoActiveGame(Exception):
    def __init__(self, gameId):
        self.gameId=gameId

class InvalidMove(Exception):
    def __init__(self, gameId, move):
        self.gameId=gameId
        self.move=move

class PlayerNotInGame(Exception):
    def __init__(self, playerId, gameId):
        self.gameId=gameId
        self.playerId=playerId

class MakeReadyError(Exception):
    def __init__(self, msg):
        self.msg=msg

class JoinGameError(Exception):
    def __init__(self, msg):
        self.msg=msg

class CPPlayer():
    def __init__(self, id, name, isAI=False):
        self.id=id
        self.name=name
        self.isAI = isAI

class CPGameManager():
    def __init__(self,seed=False):
        self.players = {}
        self.games = {}
        self.ready={}
        self.nextPlayerId=0
        self.nextGameId=0
        self.activeGames=set()
        self.pendingGame=set()
        self.seed=seed
        self.gameEnv=CPMLGameEnv.CPMLGameEnv(CPMLModelDef.num_players, CPMLModelDef.hist_len)
        self.model=keras.models.load_model(modelFileName)

    def newPlayer(self, name, isAI=False):
        newPlayerId = self.nextPlayerId
        self.players[newPlayerId]=CPPlayer(newPlayerId, name, isAI=isAI)
        self.nextPlayerId+=1
        return newPlayerId
    def getPlayerId(self, name):
        playerId=False
        for pId,pName in self.players.items():
            if name==pName[0]:
                playerId=pId
        return playerId

    def joinNextGame(self,playerId):
        if len(self.pendingGame)<4 and (playerId not in self.pendingGame):
            self.pendingGame.add(playerId)
            self.ready[playerId]=False
            return self.nextGameId
        else:
            raise JoinGameError(f"{playerId} Cant join game")
    def makeReady(self,playerId):
        if playerId not in self.ready:
            raise MakeReadyError(f"{playerId} Not in game")
        self.ready[playerId]=True
        if all(self.ready.values()) and len(self.ready.keys())>1:
            id=self.newGame(list(self.ready.keys()))
            self.ready={}
            self.pendingGame=set()
            return {'result':"game_started", "game_id":id}
        return {'result':'success'}
    def newGame(self, players):
        id=self.nextGameId
        self.nextGameId+=1
        self.games[id]=CPGame(players,seed=self.seed)
        self.activeGames.add(id)
        return id
    def implementMove(self, gameId, move):
        if gameId not in self.activeGames:
            raise NoActiveGame(gameId)
        game=self.games[gameId]
        if not game.implementMove(move):
            raise InvalidMove(gameId, move)
        if game.done():
            self.activeGames.remove(gameId)
    def runAIs(self, gameId):
        while self.players[self.games[gameId].toMove].isAI:
            game=self.games[gameId].toMove
            possibleActions = self.gameEnv.getPossibleActions(game)
            state = self.gameEnv.actionHistory(CPMLModelDef.hist_len, game)
            action_probs = CPMLModelDef.get_action_probs(self.model, possibleActions, state)
            action = np.argmax(action_probs)
            move = game.getMoves()[action]
            game.implementMove(move)
            if game.done():
                self.activeGames.remove(gameId)
        return True
    def getGameState(self, gameId):
        if gameId==self.nextGameId:
            return {'active':False,
                    'player_ready':self.ready,
                    'player_names':{k:self.players[k].name for k in self.players.keys()}
                    }
        elif gameId not in self.games:
            raise NoActiveGame(gameId)
        else:
            game=self.games[gameId]
            return {'active':True,
                    'winner': int(game.winner),
                    'to_move': int(game.toMove),
                    'last_move': {playerId:game.getLastMove(playerId)
                                  for playerId in game.players},
                    'hands': {player:sorted(list(game.hands[player])) for player in game.players},
                    'players': list(game.players),
                    'player_names':{k:self.players[k].name for k in game.players}
                    }
    def isReady(self,pId):
        return (pId in self.ready) and self.ready[pId]
    def getActiveGameByPlayer(self,pId):
        return [gId for gId,g in self.games.items() if pId in g.players]
    def playerStats(self):
        return {pId:[p.name,
                      int(p.id in self.pendingGame),
                      int(self.isReady(p.id)),
                      self.getActiveGameByPlayer(p.id)
                      ] for (pId,p) in self.players.items()}


