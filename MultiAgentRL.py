import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pdb

# Define the ResourceAllocationEnvironment
class ResourceAllocationEnvironment:
    def __init__(self, num_agents, num_resources):
        self.num_agents = num_agents
        self.num_resources = num_resources
        self.state = np.zeros((num_agents, num_resources))
        self.reward = np.zeros((num_agents))
        self.done = False

    def step(self, actions):
        # Execute actions and update the state
        # Implement your resource allocation logic here
        # Update self.state based on the chosen actions
        # Calculate reward based on your reward function
        if np.sum(self.reward) > 10:
          self.done = True
        else:
          self.done = False
          self.reward = self.reward + 1
        info = {}  # Additional information
        return self.state, self.reward, self.done, info

    def reset(self):
        # Reset the environment to the initial state
        self.state = np.zeros((self.num_agents, self.num_resources))
        self.done = False
        return self.state

# Define the Actor and Critic networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the MADDPG agent
class MADDPGAgent:
    def __init__(self, state_dim, action_dim, num_agents):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim * num_agents, action_dim * num_agents)
        self.critic_target = Critic(state_dim * num_agents, action_dim * num_agents)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

    def select_action(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            action = self.actor(state)
        return action.numpy()

    def train(self, states, actions, rewards, next_states, dones):
        # Training logic for the actor and critic networks
        pass

# Main training loop
num_agents = 2
num_resources = 5
state_dim = num_resources
action_dim = num_resources

env = ResourceAllocationEnvironment(num_agents, num_resources)
agents = [MADDPGAgent(state_dim, action_dim, num_agents) for _ in range(num_agents)]

num_episodes = 1000
for episode in range(num_episodes):
    states = env.reset()
    total_reward = 0
    done = False

    while not done:
        actions = [agent.select_action(state) for agent, state in zip(agents, states)]
        next_states, reward, done, _ = env.step(actions)
        total_reward += reward
        # Training logic for the agents
        for i, agent in enumerate(agents):
            agent.train(states, actions, [reward[i]], next_states, [done])

        states = next_states

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")


# Testing loop
num_test_episodes = 10
for episode in range(num_test_episodes):
    states = env.reset()
    total_reward = 0
    done = False

    while not done:
        actions = [agent.select_action(state) for agent, state in zip(agents, states)]
        next_states, reward, done, _ = env.step(actions)
        total_reward += reward

        states = next_states

    print(f"Testing - Episode: {episode + 1}, Total Reward: {total_reward}")

