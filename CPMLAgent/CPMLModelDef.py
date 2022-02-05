import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

num_cards = 52
hist_len = 30
num_players = 2
loss_function = keras.losses.Huber()


def create_q_model():
    # Network defined by the Deepmind paper
    model = keras.Sequential()

    model.add(layers.Input(shape=(hist_len+2,num_cards,)))
    model.add(layers.Reshape((hist_len+2,13,4)))
    model.add(layers.Permute((2,3,1)))
    model.add(layers.Conv2D(128,4, activation=layers.LeakyReLU(alpha=0.1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation=layers.LeakyReLU(alpha=0.1)))
    model.add(layers.Dense(256, activation=layers.LeakyReLU(alpha=0.1)))
    model.add(layers.Dense(1, activation=layers.LeakyReLU(alpha=0.1)))

    # inputs = layers.Input(shape=(hist_len+2,num_cards,))
    # reshape = layers.Reshape((hist_len+2,13,4))(inputs)
    # permute = layers.Permute((2,3,1))(reshape)
    # layer1 = layers.Conv2D(64,4, activation="relu")(permute)
    # layer3 = layers.Flatten()(layer1)
    # layer4 = layers.Dense(256, activation="relu")(layer3)
    # layer5 = layers.Dense(256, activation="relu")(layer4)
    # action = layers.Dense(1, activation="linear")(layer5)
    # model = keras.Model(inputs=inputs, outputs=action)
    model.compile(optimizer='adam', loss=loss_function)
    print(model.summary())
    return model


def combine_state_action(state, action):
    return np.array([[action]+state]).astype("float32")

def get_action_probs(model,possibleActions,state):
    return model.predict(np.stack([np.vstack([a, state]) for a in possibleActions]).astype('float32'))

def get_group_action_probs(model, possibleActions, states):
    states_pa=[]
    for s,pa in zip(states,possibleActions):
        states_pa += [np.vstack([a,s]) for a in pa]
    ap=model.predict(np.stack(states_pa).astype('float32'))
    k=0
    actionProbs = np.zeros(len(states))
    for i,s in enumerate(states):
        actionProbs[i] = max(ap[k:k+len(possibleActions[i])])[0]
        k+=len(possibleActions[i])
    return actionProbs