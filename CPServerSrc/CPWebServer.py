import logging
import os

from flask import Flask,request,jsonify
from cpGameManager import CPGameManager, NoActiveGameException, JoinGameError
from CPGame import MoveError


app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

gm = CPGameManager(seed=os.getenv('CP_RANDOM_SEED'))

@app.route('/newplayer/', methods=['POST'])
def apiNewPlayer():
    if not request.json or not 'name' in request.json:
        return "Could not parse newplayer", 400
    newPlayerId=gm.newPlayer(request.json['name'])
    app.logger.info(f"Created newplayer {newPlayerId} {request.json['name']}")
    return jsonify(newPlayerId), 201

@app.route('/joinnextgame/', methods=['PUT'])
def apiJoinNextyGame():
    playerId=request.json['playerid']
    try:
        joinGame=gm.joinNextGame(playerId)
    except JoinGameError as jge:
        app.logger.info(f"Join Game error {jge.msg}")
        return jge.msg, 400
    app.logger.error(f"Join game {playerId} {joinGame}")
    return jsonify(joinGame), 200


@app.route('/makeready/', methods=['PUT'])
def apiMakeReady():
    playerId=request.json['playerid']
    return jsonify(gm.makeReady(playerId))


@app.route('/implementmove/', methods=['PUT'])
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
    except:
        return "Move Error", 400

@app.route('/getgamestate/', methods=['GET'])
def apiGetGameState():
    if not request.json or not 'gameid' in request.json:
        return "Missing gameid", 400
    try:
        gameId=int(request.json['gameid'])
        return jsonify(gm.getGameState(gameId)), 200
    except NoActiveGameException(gameId):
        return "Could not find gameid", 400
if __name__ == '__main__':
    host=os.getenv('CP_HOST')
    port=os.getenv('CP_PORT')
    if host==None:
        host='0.0.0.0'
    if port==None:
        port=105
    app.run(host=host, port=port)