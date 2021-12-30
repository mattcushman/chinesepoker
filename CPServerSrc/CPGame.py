import itertools
import numpy as np

ranks = ['4','5','6','7','8','9','T','J','Q','K','A','2','3']
suits = ['d','c','h','s']

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
    if len(hand) == 5 and ( (hand[0]//4+1 == hand[1]//4) and
                            (hand[1]//4+1 == hand[2]//4) and
                            (hand[2]//4+1 == hand[3]//4) and
                            (hand[3]//4+1 == hand[4]//4)):
        sig.append(1+hand[-1])
    else:
        sig.append(0)
    if len(hand) == 5 and ( hand[0] % 4 == hand[1] % 4 and hand[1] % 4 == hand[2] % 4
                            and hand[2] % 4 == hand[3] % 4 and hand[3] % 4 == hand[4] % 4):
        sig.append(1+hand[-1])
    else:
        sig.append(0)
    return sig

def subsets(k, l):
    return [list(x) for x in itertools.combinations(l,k)]

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

    def cardToString(self, c):
        return ranks[c // 4] + suits[c % 4]

    def cardsToString(self, cards):
        return "-".join([self.cardToString(c) for c in sorted(cards)])

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
            raise MoveError(f"Not valid move signature={moveSignature}", move)
        if lastRealPlayer==self.toMove or all([x>=y for (x,y) in zip(moveSignature, computeMoveSignature(lastRealMove))]):
            self.doMove(move)
            return True
        else:
            print(moveSignature, computeMoveSignature(lastRealMove))
            raise MoveError("Move does not beat last move", move)
    def doMove(self, move):
        for c in move:
            self.hands[self.toMove].remove(c)
        self.playerMoves.append([self.toMove, move])
        if len(self.hands[self.toMove])==0:
            self.winner = self.toMove
        self.toMove = self.players[(self.players.index(self.toMove)+1) % len(self.players)]

    def getLastMove(self, playerId):
        lm=[m for (p,m) in self.playerMoves if p==playerId]
        if lm==[]:
            return []
        else:
            return lm[-1]

    def allMoves(self, cards):
        moves = [[c] for c in cards]
        for rnk in range(13):
            cardsOfRank = [c for c in cards if c//4==rnk]
            moves = moves + subsets(2, cardsOfRank)
            moves = moves + subsets(3, cardsOfRank)
            moves = moves + subsets(4, cardsOfRank)
            if (rnk+4<13) and (len(cardsOfRank)>0):
                straights= [ [c] for c in cardsOfRank]
                for k in range(1,5):
                    straights = [ s + [c] for s in straights for c in cards if c//4==rnk+k]
                moves = moves + straights
        for pair in moves:
            for trip in moves:
                if len(pair)==2 and len(trip)==3 and pair[0]//4 != trip[0]//3:
                    moves.append(pair+trip)
        for suit in range(4):
            cardsOfSuit = [c for c in cards if c%4==suit]
            moves = moves + subsets(5, cardsOfSuit)
        return moves

    def getMoves(self):
        allMoves = self.allMoves(self.hands[self.toMove])
        if len(self.playerMoves)==0:
            c=min(self.hands[self.toMove])
            return [m for m in allMoves if c in m]
        (lastRealPlayer, lastRealMove) = [pm for pm in self.playerMoves if len(pm[1]) > 0][-1]
        if self.toMove==lastRealPlayer:
            return allMoves
        else:
            lastMoveSignature = computeMoveSignature(lastRealMove)
            print("HERE!!!  {computeMoveSignature(m)} {lastMoveSignature}")
            return [ [] ] + [m for m in allMoves if all([x>=y for (x,y) in zip(computeMoveSignature(m), lastMoveSignature)])]

    def done(self):
        return any([len(hand)==0 for hand in self.hands.values()])

    def prettyState(self):
        str = " | ".join(f"p={p} hand={self.cardsToString(self.hands[p])}" for p in self.players)
        if self.playerMoves == []:
            return str
        else:
            return f"player={self.playerMoves[-1][0]} move={self.cardsToString(self.playerMoves[-1][1])} / "+str
