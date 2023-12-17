import numpy as np
import gym
import pdb

# Create the FrozenLake environment
env = gym.make('FrozenLake-v1')

# Q-table initialization
num_states = env.observation_space.n
num_actions = env.action_space.n
Q_table = np.zeros((num_states, num_actions))

# Parameters
learning_rate = 0.8
discount_factor = 0.95
num_episodes = 1000

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()[0]
    done = False
    while not done:
        # Choose an action using epsilon-greedy policy
        if np.random.rand() < 0.3:
            action = env.action_space.sample()  # Exploration
        else:
            action = np.argmax(Q_table[state, :])  # Exploitation

        # Take the chosen action and observe the next state and reward
        next_state, reward, done, info, _ = env.step(action)

        # Update the Q-value using the Q-learning update rule
        Q_table[state, action] = (1 - learning_rate) * Q_table[state, action] + learning_rate * (reward + discount_factor * np.max(Q_table[next_state, :]))

        # Move to the next state
        state = next_state

print(Q_table)
pdb.set_trace()
# Evaluate the learned policy
total_reward = 0
num_eval_episodes = 10

for _ in range(num_eval_episodes):
    state = env.reset()[0]
    done = False

    while not done:
        action = np.argmax(Q_table[state, :])
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state

average_reward = total_reward / num_eval_episodes
print(f"Average reward over {num_eval_episodes} episodes: {average_reward}")

# Close the environment
env.close()
