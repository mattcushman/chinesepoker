from CPMLAgent import CPMLGameEnv
import random

g=CPMLGameEnv.CPMLGameEnv(2)

g.reset()

while not g.done():
    print(g.prettyState())
    move=random.choice(g.getPossibleActions())
    print(f"move={move}")
    state,reward,done,msg = g.step(move)
    print(f"reward={reward}")
    print('---------------------------------------------------------------------')
