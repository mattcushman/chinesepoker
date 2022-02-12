import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

num_cards = 52
hist_len = 64
num_players = 2
loss_function = keras.losses.Huber()


def create_q_model():
    # model = keras.Sequential()
    # activation='relu'
    # model.add(layers.Input(shape=(hist_len+2,num_cards,)))
    # model.add(layers.Reshape((hist_len+2,13,4)))
    # model.add(layers.Permute((2,3,1)))
    # model.add(layers.Conv2D(64,4, activation=activation))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(256, activation=activation))
    # model.add(layers.Dense(256, activation=activation))
    # model.add(layers.Dense(1, activation=activation))

    hand_layer1 = layers.Conv1D(16,4,4, name="hand_layer1", activation="relu")
    hand_layer2 = layers.Dense(32, name="hand_layer2", activation="relu")

    hist_input = layers.Input(shape=(hist_len,num_cards,1,), name="history")
    hist_inter1 = hand_layer2(hand_layer1(hist_input))
    hist_inter2 = layers.Conv2D(32,4, strides=2, activation="relu")(hist_inter1)
    hist_output = layers.Flatten()(hist_inter2)

    current_hand_input = layers.Input((num_cards,), name="hand")
    current_hand_output = layers.Dense(64, activation="relu")(current_hand_input)

    move_input = layers.Input((num_cards,), name="move")
    move_output = layers.Dense(64, activation="relu")(move_input)

    x = layers.concatenate([move_output, current_hand_output, hist_output])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    action = layers.Dense(1, activation="linear")(x)
    model = keras.Model(inputs=[move_input, current_hand_input, hist_input], outputs=action)
    model.compile(optimizer='adam', loss=loss_function)
    print(model.summary())
    return model


def get_action_probs(model,possibleActions,state):
    return model.predict([np.array(possibleActions),
                          np.array([state[0]]*len(possibleActions)),
                          np.array([state[1:]]*len(possibleActions))]).astype('float32')

def get_group_action_probs(model, possibleActions, states):
    hl = max([h.shape[0]-1 for h in states])
    actions = []
    hands = []
    histories = []
    for s,pa in zip(states,possibleActions):
        actions += pa
        hands += [s[0]] * len(pa)
        histories += [s[1:]] * len(pa)
    ap=model.predict([np.array(actions),
                      np.array(hands),
                      np.array(histories)]).astype('float32')
    k=0
    actionProbs = np.zeros(len(states))
    for i,s in enumerate(states):
        actionProbs[i] = max(ap[k:k+len(possibleActions[i])])[0]
        k+=len(possibleActions[i])
    return actionProbs