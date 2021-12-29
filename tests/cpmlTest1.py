from CPMLAgent import CPMLGameEnv
import random
import numpy as np

g=CPMLGameEnv.CPMLGameEnv(2,15)

g.reset()

while not g.done():
    print(g.prettyState())
    move=np.random.choice(len(g.getPossibleActions()))
    print(f"move={move}")
    state,reward,done,msg = g.step(move)
    print(f"reward={reward}")
    print('---------------------------------------------------------------------')

print(g.prettyState())