import logging

from flask import Flask,request,jsonify
from cpGameManager import CPGameManager, NoActiveGameException


app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

gm = CPGameManager(seed=12345)

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
    joinGame=gm.joinNextGame(playerId)
    app.logger.info(f"Join game {playerId} {joinGame}")
    return jsonify(joinGame)


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
    app.run(host='0.0.0.0', port=105)