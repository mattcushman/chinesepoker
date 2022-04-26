import json
import os.path
import hashlib
import random

from CPGame import CPGame
from CPMLAgent import CPMLModelDef
from CPMLAgent import CPMLGameEnv
from tensorflow import keras
import numpy as np

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
    def __init__(self, id, name, pw, isAI=False, modelName="cpmlModel1"):
        self.id=id
        self.name=name
        self.pw=pw
        self.isAI = isAI
        self.modelName = modelName
        self.token = False
    def login(self, pw):
        if pw==self.pw:
            self.token = hashlib.sha256(str(random.randint(0,2**128)).encode('utf-8')).hexdigest()
            return self.id,self.token
        else:
            return False,0
class CPGameManager():
    def __init__(self, seed=False, historyFile="cphistory.json"):
        self.players = {}
        self.games = {}
        self.ready={}
        self.nextPlayerId=0
        self.nextGameId=0
        self.gamehistory=[]
        self.activeGames=set()
        self.pendingGame=set()
        self.seed=seed
        self.historyFile=historyFile
        self.gameEnv={}

        if os.path.exists(self.historyFile):
            hFileHandle = open(historyFile)
            historyData = json.load(hFileHandle)
            self.players={pid:CPPlayer(pid, name, pw, isAI, modelName) for [pid,name, pw, isAI, modelName] in historyData['players']}
            self.gameHistory=historyData['gamehistory']
            self.nextPlayerId=historyData['nextPlayerId']
            self.nextGameId = historyData['nextGameId']

        for pId in self.players.keys():
            if self.players[pId].isAI:
                self.gameEnv[pId] = CPMLGameEnv.CPMLGameEnv(CPMLModelDef.num_players, CPMLModelDef.hist_len)

        mnames = ["cpmlModel0", "cpmlModel1"]
        self.models={}
        for mname in mnames:
            self.models[mname]=keras.models.load_model(f"../CPTextGame/{mname}")

    def saveHistory(self):
        hFileHandle=open(self.historyFile, "w")
        json.dump({
            "players":[[p.id,p.name,p.pw,int(p.isAI), p.modelName] for pid,p in self.players.items()],
            "nextPlayerId":self.nextPlayerId,
            "gamehistory":self.gamehistory,
            "nextGameId":self.nextGameId
        },
                  hFileHandle,indent=4)
    def newPlayer(self, name, pw, isAI=False, modelName=None):
        newPlayerId = self.nextPlayerId
        self.players[newPlayerId]=CPPlayer(newPlayerId, name, pw, isAI=isAI, modelName=modelName)
        self.nextPlayerId+=1
        self.saveHistory()
        return newPlayerId
    def login(self,name,pw):
        player = self.getPlayer(name)
        if player:
            return player.login(pw)
        else:
            return False,0

    def getPlayer(self, name):
        namedPlayer=False
        for pId,player in self.players.items():
            if name==player.name:
                namedPlayer=player
        return namedPlayer

    def getPlayerId(self,name):
        p=self.getPlayer(name)
        if p:
            return p.id
        else:
            return False

    def getPlayersToken(self,pid):
        for pId,player in self.players.items():
            if pId==pid:
                return player.token
        return False

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

    def startAiGame(self,playerId,aiPlayerId):
        if playerId not in self.ready and getActiveGameByPlayer(playerId)==[] and getActiveGameByPlayer(aiPlayerId)==[]:
            gameId=self.nextGame([playerId, aiPlayerId])
            self.runAIs(gameId)
            return gameId
        else:
            return -1

    def implementMove(self, gameId, move):
        if gameId not in self.activeGames:
            raise NoActiveGame(gameId)
        game=self.games[gameId]
        if not game.implementMove(move):
            raise InvalidMove(gameId, move)
        self.runAIs(gameId)
        if game.done():
            self.activeGames.remove(gameId)
            self.saveHistory()
        return 1

    def runAIs(self, gameId):
        game = self.games[gameId]
        while not game.done() and self.players[game.toMove].isAI:
            possibleActions = self.gameEnv[game.toMove].getPossibleActions(game)
            state = self.gameEnv[game.toMove].actionHistory(CPMLModelDef.hist_len, game)
            model = self.models[self.players[game.toMove].modelName]
            action_probs = CPMLModelDef.get_action_probs(model, possibleActions, state)
            action = np.argmax(action_probs)
            move = game.getMoves()[action]
            game.implementMove(move)
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
        return [gId for gId,g in self.games.items() if pId in g.players and gId in self.activeGames]

    def playerStats(self):
        return {pId:[p.name,
                      int(p.id in self.pendingGame),
                      int(self.isReady(p.id)),
                      int(p.isAI),
                      self.getActiveGameByPlayer(p.id)
                      ] for (pId,p) in self.players.items()}


