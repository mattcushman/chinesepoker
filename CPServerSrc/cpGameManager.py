from CPGame import CPGame

class NoActiveGameException(Exception):
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
    def __init__(self, id, name):
        self.id=id
        self.name=name


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
    def newPlayer(self, name):
        newPlayerId = self.nextPlayerId
        self.players[newPlayerId]=CPPlayer(newPlayerId, name)
        self.nextPlayerId+=1
        return newPlayerId
    def joinNextGame(self,playerId):
        if len(self.pendingGame)<4 and (playerId not in self.pendingGame):
            self.pendingGame.add(playerId)
            self.ready[playerId]=False
            return True
        else:
            raise GameJoinError(f"{playerId} Cant join game")
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
        return True
    def getGameState(self, gameId):
        if gameId not in self.games:
            raise NoActiveGameException(gameId)
        else:
            game=self.games[gameId]
            return {'winner': int(game.winner),
                    'to_move': int(game.toMove),
                    'last_move': {playerId:game.getLastMove(playerId)
                                  for playerId in game.players},
                    'hands': {player:sorted(list(game.hands[i])) for i,player in enumerate(game.players)},
                    'players': list(game.players)
                    }



