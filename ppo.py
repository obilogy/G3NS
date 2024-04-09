import torch
import torch.nn as nn
import numpy as np



class PPOPolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(PPOPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 64)
        self.action_head = nn.Linear(64, num_actions)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        #print(f"Input to fc1: {x}")  # Check input values
        x = torch.relu(self.fc1(x))
        if torch.isnan(x).any():
            print(f"NaN detected after fc1: {x}")
        x = torch.relu(self.fc2(x))
        if torch.isnan(x).any():
            print(f"NaN detected after fc2: {x}")
        action_probs = torch.softmax(self.action_head(x), dim=-1)
        if torch.isnan(action_probs).any():
            print(f"NaN detected in action_probs: {action_probs}")
        state_values = self.value_head(x)
        return action_probs, state_values



class PPO:
    def __init__(self, policy_network, optimizer):
        self.policy_network = policy_network
        self.optimizer = optimizer

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, _ = self.policy_network(state)
        action = np.random.choice(len(probs.detach().numpy()[0]), p=probs.detach().numpy()[0])
        return action

    def compute_returns(self, rewards, masks, gamma=0.99):
        returns = []
        R = 0
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def update(self, states, actions, log_probs_old, returns, advantages, epsilon=0.1, c1=1.0, c2=0.01):
        #states = torch.tensor(states, dtype=torch.float32)
        states = states.clone().detach()
        actions = torch.tensor(actions)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        for _ in range(4):  # update policy for several epochs
            log_probs, state_values = self.evaluate_actions(states, actions)
            ratios = torch.exp(log_probs - log_probs_old)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon) * advantages
            action_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns - state_values).pow(2).mean()
            loss = action_loss + c1 * value_loss - c2 * (log_probs * log_probs).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def evaluate_actions(self, states, actions):
        probs, state_values = self.policy_network(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        return log_probs, state_values.squeeze(-1)

