"""
Title: Deep Q-Learning for Atari Breakout
Author: [Jacob Chapman](https://twitter.com/jacoblchapman) and [Mathias Lechner](https://twitter.com/MLech20)
Date created: 2020/05/23
Last modified: 2020/06/17
Description: Play Atari Breakout with a Deep Q-Network.
"""
"""
## Introduction

This script shows an implementation of Deep Q-Learning on the
`BreakoutNoFrameskip-v4` environment.

This example requires the following dependencies: `baselines`, `atari-py`, `rows`.
They can be installed via:

```
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
git clone https://github.com/openai/atari-py
wget http://www.atarimania.com/roms/Roms.rar
unrar x Roms.rar .
python -m atari_py.import_roms .
```

### Deep Q-Learning

As an agent takes actions and moves through an environment, it learns to map
the observed state of the environment to an action. An agent will choose an action
in a given state based on a "Q-value", which is a weighted reward based on the
expected highest long-term reward. A Q-Learning Agent learns to perform its
task such that the recommended action maximizes the potential future rewards.
This method is considered an "Off-Policy" method,
meaning its Q values are updated assuming that the best action was chosen, even
if the best action was not chosen.

### Atari Breakout

In this environment, a board moves along the bottom of the screen returning a ball that
will destroy blocks at the top of the screen.
The aim of the game is to remove all blocks and breakout of the
level. The agent must learn to control the board by moving left and right, returning the
ball and removing all the blocks without the ball passing the board.

### Note

The Deepmind paper trained for "a total of 50 million frames (that is, around 38 days of
game experience in total)". However this script will give good results at around 10
million frames which are processed in less than 24 hours on a modern machine.

### References

- [Q-Learning](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf)
- [Deep Q-Learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning)
"""
"""
## Setup
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import CPMLGameEnv


# Configuration paramaters for the whole setup
seed = 42
gamma = 0.98  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.05  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000

num_cards = 52
hist_len = 30
num_players = 2

# Use the Baseline Atari environment because of Deepmind helper functions
env = CPMLGameEnv.CPMLGameEnv(num_players, hist_len)
loss_function = keras.losses.Huber()

"""
## Implement the Deep Q-Network

This network learns an approximation of the Q-table, which is a mapping between
the states and actions that an agent will take. For every state we'll have four
actions, that can be taken. The environment provides the state, and the action
is chosen by selecting the larger of the four Q-values predicted in the output layer.

"""

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(hist_len+2, num_cards,1 ))

    layer1 = layers.Dense(32, activation="relu")(inputs)
    layer2 = layers.Dense(64, activation="relu")(layer1)
    layer3 = layers.Dense(64, activation="relu")(layer2)
    layer4 = layers.Dense(128, activation="relu")(layer3)
    layer5 = layers.Flatten()(layer4)
    action = layers.Dense(1, activation="linear")(layer5)
    model = keras.Model(inputs=inputs, outputs=action)
    model.compile(optimizer='adam', loss=loss_function)
    return model


# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()


"""
## Train
"""
# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
done_history = []
possible_actions = []
rewards_history = []
running_reward = 0
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output
epsilon_random_frames = 1000
# Number of frames for exploration
epsilon_greedy_frames = 100000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 10000


def combine_state_action(state, action):
    return np.array([[action]+state]).astype("float32")

def get_action_probs(model,possibleActions,state):
    return model.predict(np.expand_dims(np.stack([np.vstack([a, state]) for a in possibleActions]).astype('float32'),axis=3))


while True:  # Run until solved
    state = np.array(env.reset())

    for timestep in range(1, max_steps_per_episode):
        if frame_count%10==0:
            print(f'frame count={frame_count}')

        do_print = (frame_count % 1000)<20
        if do_print:
            print(env.prettyState())
        frame_count += 1

        # Use epsilon-greedy for exploration
        possibleActions = env.getPossibleActions()

        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(len(possibleActions))
            if do_print:
                print(f"Random action {epsilon}")
        else:
            # Predict action Q-values
            # From environment state
            action_probs = get_action_probs(model, possibleActions, state)
            # Take best action
            action = np.argmax(action_probs)
            if do_print:
                print(f"Best action {action}")
                for i,(hand,prob) in enumerate(zip(env.game.getMoves(), action_probs.tolist())):
                     print(f"({i}) {env.game.cardsToString(hand):>15} {prob[0]:8.6f}")

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done, _ = env.step(action)
        state_next = np.array(state_next)

        # Save actions and states in replay buffer
        action_history.append(possibleActions[action])
        possible_actions.append(possibleActions)
        state_history.append(state)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)-num_players), size=batch_size)

            # Using list comprehension to sample from replay buffer
            # state_sample = np.array([state_history[i] for i in indices]).astype("float32")
            state_next_sample = np.array([state_history[i+num_players] for i in indices]).astype("float32")
            state_action_sample = np.array([np.vstack([np.array(action_history[i]),state_history[i]]) for i in indices]).astype("float32")
            state_action_next_sample = np.stack([np.vstack([np.array(action_history[i+num_players]),state_history[i+num_players]]) for i in indices])
            action_sample = [action_history[i] for i in indices]
            done_sample = np.array([any([done_history[i+j] for j in range(min(num_players, len(done_history)-i))]) for i in indices]).astype("float32")
            rewards_sample = np.array([rewards_history[i] for i in indices]).astype("float32")
            future_rewards = np.array([ max(get_action_probs(model_target,
                                                              possible_actions[i+num_players],
                                                              state_history[i+num_players]))[0]  for i in indices]).astype("float32")
            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            # future_rewards = model_target.predict(np.expand_dims(state_action_next_sample,axis=3))
            # Q value = reward + discount factor * expected future reward
            updated_q_values = np.expand_dims(done_sample*rewards_sample,axis=1) + gamma * (1-np.expand_dims(done_sample,axis=1))*future_rewards

            # Create a mask so we only calculate loss on the updated Q-values
          #  masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(np.expand_dims(state_action_sample,axis=3))

                # Apply the masks to the Q-values to get the Q-value for action taken
     #           q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_values)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            print("Update target network")
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))
            print("Saving model")
            model.save('cpmlModel')

        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del action_history[:1]
            del possible_actions[:1]
            del done_history[:1]

        if done:
            break


