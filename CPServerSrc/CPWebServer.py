import logging
import os

from flask import Flask,request,jsonify
from flask_cors import CORS, cross_origin
from cpGameManager import CPGameManager, NoActiveGame, JoinGameError, InvalidMove
from CPGame import MoveError


app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def initGame():
    gm = CPGameManager(seed=os.getenv('CP_RANDOM_SEED'))
    return gm

gm=initGame()

@app.route('/newplayer/', methods=['POST'])
@cross_origin()
def apiNewPlayer():
    if not request.json or not 'name' in request.json:
        return "Could not parse newplayer", 400
    playerId = gm.getPlayerId(request.json['name'])
    if not playerId:
        playerId=gm.newPlayer(request.json['name'])
        app.logger.info(f"Created newplayer {playerId} {request.json['name']}")
    return jsonify(playerId), 201

@app.route('/playerstats/', methods=['POST'])
@cross_origin()
def apiPlayerStats():
    playerStats=gm.playerStats()
    return jsonify(playerStats), 201

@app.route('/joinnextgame/', methods=['PUT'])
@cross_origin()
def apiJoinNextyGame():
    playerId=request.json['playerid']
    try:
        joinGame=gm.joinNextGame(playerId)
    except JoinGameError as jge:
        app.logger.info(f"Join Game error {jge.msg}")
        return jge.msg, 400
    app.logger.info(f"Join game {playerId} {joinGame}")
    return jsonify(joinGame), 200


@app.route('/makeready/', methods=['PUT'])
@cross_origin()
def apiMakeReady():
    playerId=request.json['playerid']
    return jsonify(gm.makeReady(playerId))


@app.route('/implementmove/', methods=['PUT'])
@cross_origin()
def apiImplementMove():
    if not request.json or not 'gameid' in request.json or not 'move' in request.json:
        return "Not formatted correctly", 400
    try:
        gameId=int(request.json['gameid'])
        move=list(request.json['move'])
        return jsonify(gm.implementMove(gameId,move)), 200
    except MoveError as merr:
        app.logger.error(f"MoveError {merr.move} {merr.msg}")
        return f"MoveError {merr.move} {merr.msg}", 406
    except InvalidMove as merr:
        app.logger.error(f"InvalidMove {merr.move} {merr.gameId}")
        return f"Invalid Move {merr.move} {merr.gameId}", 406
    except NoActiveGame as nag:
        app.logger.error(f"NoActiveGame(gameId={nag.gameId})")
        return f"NoActiveGameException(gameId={gameId})", 400
    except:
        return "Other Error", 400

@app.route('/getgamestate/', methods=['POST'])
@cross_origin()
def apiGetGameState():
    if not request.json or not 'gameid' in request.json:
        return "Missing gameid", 400
    try:
        gameId=int(request.json['gameid'])
        return jsonify(gm.getGameState(gameId)), 200
    except NoActiveGame as nag:
        return f"NoActiveGameException(gameId={nag.gameId})", 400

@app.route('/startaigame/', methods=['PUT'])
@cross_origin()
def startAIGame():
    playerId=request.json['playerid']
    app.logger.info(f"Started AI game with {playerId}")
    return jsonify(gm.startAiGame(playerId))


if __name__ == '__main__':
    host=os.getenv('CP_HOST')
    port=os.getenv('CP_PORT')
    if host==None:
        host='0.0.0.0'
    if port==None:
        port=105
    app.run(host=host, port=port)