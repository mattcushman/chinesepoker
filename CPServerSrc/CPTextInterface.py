from cpGameManager import CPGameManager
import fileinput

gm = CPGameManager()

for cmd in fileinput.input():
    if cmd.startswith("newPlayer"):
        print(gm.newPlayer(cmd[10:]))
    elif cmd.startswith("joinNextGame"):
        print(gm.joinNextGame(int(cmd[12:])))
    elif cmd.startswith("makeReady"):
        print(gm.makeReady(int(cmd[9:])))
    elif cmd.startswith("implementMove"):
        parsedArgs=[int(i) for i in cmd[13:].split(" ")]
        print(gm.implementMove(gameId, parsedArgs[0], parsedArgs[1:]))
    elif cmd.startswith("getGameState"):
        print(gm.getGameState(int(cmd[12:])))







