import numpy as np

class MoveError(Exception):
    def __init__(self, msg, move):
        self.msg=msg
        self.move=move

def computeMoveSignature(hand):
    sig=[]
    if len(hand)==1:
        sig.append(1+hand[-1])
    else:
        sig.append(0)
    if len(hand)==2 and (hand[0]//4==hand[1]//4):
        sig.append(1+hand[-1])
    else:
        sig.append(0)
    if len(hand) == 3 and (hand[0] // 4 == hand[1] // 4 and hand[1]//4==hand[2]//4):
        sig.append(1+hand[-1])
    else:
        sig.append(0)
    if len(hand) == 4 and (hand[0] // 4 == hand[1] // 4 and hand[1] // 4 == hand[2] // 4 and hand[2] // 4 == hand[3] // 4):
        sig.append(1+hand[-1])
    else:
        sig.append(0)
    if len(hand) == 5 and ( hand[0] // 4 == hand[1] // 4 and hand[1] // 4 == hand[2] // 4 and hand[3] // 4 == hand[4] // 4):
        sig.append(1+hand[2])
    elif len(hand) == 5 and ( hand[0] // 4 == hand[1] // 4 and hand[2] // 4 == hand[3] // 4 and hand[3] // 4 == hand[4] // 4):
        sig.append(1+hand[-1])
    else:
        sig.append(0)
    if len(hand) == 5 and ( (hand[0]//4+1== hand[1]//4) and
                            (hand[1]//4+1== hand[2]//4) and
                            (hand[2]//4+1== hand[3]//4) and
                            (hand[3]//4+1== hand[4]//4)):
        sig.append(1+hand[-1])
    else:
        sig.append(0)
    if len(hand) == 5 and ( hand[0] % 4 == hand[1] % 4 and hand[1] % 4 == hand[2] % 4
                            and hand[2] % 4 == hand[3] % 4 and hand[3] % 4 == hand[4] % 4):
        sig.append(1+hand[-1])
    else:
        sig.append(0)
    return sig

class CPGame():
    def __init__(self, players,seed=False):
        if seed:
            np.random.seed(seed)
        self.players=players
        self.deck=[int(x) for x in np.random.permutation(52)]
        self.hands={}
        for i,playerId in enumerate(players):
            self.hands[playerId]=set(self.deck[13*i:13*i+13])
        self.toMove = players[np.argmin([min(self.hands[playerId]) for playerId in players])]
        self.playerMoves=[]
        self.winner=-1
    def implementMove(self, move):
        if not all(c in self.hands[self.toMove] for c in move):
            raise MoveError("Not all cards are in hand", move)
        if len(self.playerMoves)==0:
            if not min(self.hands[self.toMove]) in move:
                raise MoveError("Move doesnt contain smallest card", move)
            (lastRealPlayer, lastRealMove)=(self.toMove,[])
        else:
            (lastRealPlayer,lastRealMove) = [pm for pm in self.playerMoves if len(pm[1])>0][-1]
        if len(move)==0 and (lastRealPlayer!=self.toMove):
            self.doMove([])
            return True
        moveSignature = computeMoveSignature(move)
        if not sum(moveSignature)>0:
            raise MoveError("Not valid move", move)
        if lastRealPlayer==self.toMove or all([x>=y for (x,y) in zip(moveSignature, computeMoveSignature(lastRealMove))]):
            self.doMove(move)
            return True
        else:
            raise MoveError("Move does not beat last move", move)
    def doMove(self, move):
        for c in move:
            self.hands[self.toMove].remove(c)
        self.playerMoves.append([self.players[self.toMove], move])
        if len(self.hands[self.toMove])==0:
            self.winner = self.players[self.toMove]
        self.toMove = self.players[self.players.index(self.toMove)+1 % len(self.players)]

    def getLastMove(self, playerId):
        l=[m for (p,m) in self.playerMoves if p==playerId]
        if l==[]:
            return []
        else:
            return l[-1]
    def done(self):
        print(self.hands)
        return any([len(hand)==0 for hand in self.hands.values()])
