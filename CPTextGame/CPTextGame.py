import numpy as np
import tensorflow as tf
from tensorflow import keras
from CPMLAgent import CPMLGameEnv
from CPMLAgent.CPMLModelDef import get_action_probs, num_players, hist_len

modelFileName = "./cpmlModel0"

print(keras.__version__)

model = keras.models.load_model(modelFileName)
print(model.summary())

env = CPMLGameEnv.CPMLGameEnv(num_players, hist_len)

done=True
while True:
    if done:
        state = np.array(env.reset())

    possibleActions = env.getPossibleActions()

    if env.game.toMove==0:
        print(f"Current hand: {env.game.cardsToString(env.game.hands[0])}")
        for i, hand in enumerate(env.game.getMoves()):
            print(f"({i:2}) {env.game.cardsToString(hand):>15}")
        action = -1
        while action<0 or action>=len(possibleActions):
            action=int(input("Your move? "))
        move=possibleActions[action]
    else:
        action_probs = get_action_probs(model, possibleActions, state)
        action = np.argmax(action_probs)
        hand = env.game.getMoves()[action]
        print(f"Player {env.game.toMove} move is [{env.game.cardsToString(hand)}]")

    state, reward, done, _ = env.step(action)
    if reward==1:
        print(f"Game Over!")





