import itertools
import numpy as np

ranks = ['4','5','6','7','8','9','T','J','Q','K','A','2','3']
suits = ['\u2666','\u2663','\u2665','\u2660']

def cardToString(c):
    return ranks[c // 4] + suits[c % 4]

def cardsToString(cards):
    return "-".join([cardToString(c) for c in sorted(cards)])

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
    def __init__(self, players,seed=False,deck=None):
        if seed:
            np.random.seed(seed)
        self.players=players
        if deck is None:
            self.deck=[int(x) for x in np.random.permutation(52)]
        else:
            self.deck=deck
        self.hands={}
        for i,playerId in enumerate(players):
            self.hands[playerId]=set(self.deck[13*i:13*i+13])
        self.toMove = players[np.argmin([min(self.hands[playerId]) for playerId in players])]
        self.playerMoves=[]
        self.winner=-1

    def reward(self, player_name):
        if self.winner == -1:
            return 0
        elif self.winner == player_name:
            return 1
        else:
            return -1 
    
    def to_move_index(self):
        return self.players.index(self.toMove)

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
                if len(pair)==2 and len(trip)==3 and pair[0]//4 != trip[0]//4:
                    moves.append(pair+trip)
        for suit in range(4):
            cardsOfSuit = [c for c in cards if c%4==suit]
            moves = moves + subsets(5, cardsOfSuit)
        return moves

    def getMoves(self, player=None):
        if player is not None and player!=self.toMove:
            return []
        allMoves = self.allMoves(self.hands[self.toMove])
        if len(self.playerMoves)==0:
            c=min(self.hands[self.toMove])
            return [m for m in allMoves if c in m]
        (lastRealPlayer, lastRealMove) = [pm for pm in self.playerMoves if len(pm[1]) > 0][-1]
        if self.toMove==lastRealPlayer:
            return allMoves
        else:
            lastMoveSignature = computeMoveSignature(lastRealMove)
            moves = [ [] ]
            for m in allMoves:
                if all([x>=y for (x,y) in zip(computeMoveSignature(m), lastMoveSignature)]):
                    moves.append(m)
            return moves

    def done(self):
        return any([len(hand)==0 for hand in self.hands.values()])

    def prettyState(self):
        str = "\n".join(f'player={p} {[" ","*"][p==self.toMove]} hand={cardsToString(self.hands[p])}' for p in self.players)
        if self.playerMoves == []:
            return f"Start: \n{str}"
        else:
            return f"player={self.playerMoves[-1][0]} move=[{cardsToString(self.playerMoves[-1][1])}] \n"+str
        
    def pretty_print_game(self):
        output = []
        current_hands = self.hands.copy()
        to_move = self.toMove
        for (i, (player, move)) in reversed(list(enumerate(self.playerMoves))):
            assert current_hands[player].isdisjoint(set(move))
            current_hands[player] = current_hands[player].union(set(move))
            output += [f"player={player} move=[{cardsToString(move)}] hand=[{cardsToString(current_hands[player])}]"]
        return reversed(output)

