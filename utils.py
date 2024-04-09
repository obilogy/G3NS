from environment import SimulatedEnvironment
from controller import NetworkController
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np

def get_state_representation(current_output, desired_output, num_actions):
    # Compute the ratio of current output to desired output
    ratio = np.divide(current_output, desired_output, out=np.zeros_like(current_output), where=desired_output!=0)
    flattened_ratio = ratio.flatten()
    max_size = num_actions - 1  # Adjust based on your policy network's input size
    if len(flattened_ratio) > max_size:
        flattened_ratio = flattened_ratio[:max_size]
    state = np.zeros(num_actions)
    state[1:] = flattened_ratio  # Skip the first element for the action index
    return state

def train_with_environment(network, ppo_agent, epochs, num_samples):
    environment = SimulatedEnvironment(network)
    controller = NetworkController(network)
    optimizer = torch.optim.RMSprop(network.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    writer = SummaryWriter()
    possible_actions = network.define_possible_actions()

    for epoch in range(epochs):
        network.train()
        total_loss = 0.0
        total_performance_metric = 0.0

        states, actions, rewards, next_states, masks = [], [], [], [], []

        for i in range(num_samples):
            network.reset_F_v()

            sensory_input = environment.generate_sensory_input()
            network.set_external_inputs(sensory_input, 'cpu')

            # Generate non-linear target output based on sensory inputs
            weighted_sum_input = sum(input_value ** 2 * random.uniform(0.5, 1.5) for input_value in sensory_input.values())
            desired_motor_output = [weighted_sum_input / len(network.sensory_neurons) for _ in range(len(network.motor_neurons))]
            desired_motor_output = [output + np.random.normal(0, 0.05) for output in desired_motor_output]
            desired_motor_output_tensor = torch.tensor(desired_motor_output, dtype=torch.float32).unsqueeze(-1).repeat(1, network.state_vector_length)

            network.forward(timesteps=1)
            motor_output = network.get_motor_neuron_outputs()

            # Convert motor outputs to numpy array for state representation
            motor_output_np = np.array([output.detach().numpy() for output in motor_output.values()]).flatten()

            # Update state representation
            state = get_state_representation(motor_output_np, desired_motor_output_tensor.numpy().flatten(), len(possible_actions))

            action_idx = ppo_agent.select_action(state)
            action = possible_actions[action_idx]
            network.modify_F_v(action)

            motor_output_tensor = torch.stack([motor_output[neuron].detach().requires_grad_() for neuron in network.motor_neurons])
            loss = loss_function(motor_output_tensor, desired_motor_output_tensor)
            optimizer.zero_grad()
            loss.backward()
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            performance_metric = torch.mean((desired_motor_output_tensor - motor_output_tensor) ** 2)
            total_performance_metric += performance_metric.item()

            # Log neuron activities for each sample
            for i, neuron_state in enumerate(network.neuron_states):
                writer.add_histogram(f'Neuron_{i}_Activity', neuron_state.detach().cpu().numpy(), global_step=epoch * num_samples + i)

            reward = -performance_metric.item()
            next_state = get_state_representation(motor_output_np, desired_motor_output_tensor.numpy().flatten(), len(possible_actions))
            done = False

            states.append(state)
            actions.append(action_idx)
            rewards.append(reward)
            next_states.append(next_state)
            masks.append(1.0 - done)

            # Log action and reward
            action_details = possible_actions[action_idx]
            writer.add_text('Action/Details', str(action_details), epoch * num_samples + i)
            writer.add_scalar('Reward/Received', reward, epoch * num_samples + i)
            writer.add_scalar('Action/Chosen', action_idx, epoch * num_samples + i)


        returns = ppo_agent.compute_returns(rewards, masks)
        advantages = [returns[i] - rewards[i] for i in range(len(rewards))]

        states = np.array(states, dtype=np.float32)
        log_probs_old = []
        for state, action in zip(states, actions):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            probs, _ = ppo_agent.policy_network(state_tensor)
            if torch.isnan(probs).any():
                print(f"NaN detected in probs: {probs}")
                continue
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(torch.tensor(action))
            log_probs_old.append(log_prob.item())

        log_probs_old_tensor = torch.tensor(log_probs_old, dtype=torch.float32)

        ppo_agent.update(torch.from_numpy(states), actions, log_probs_old_tensor, returns, advantages)

        average_loss = total_loss / num_samples
        average_performance_metric = total_performance_metric / num_samples

        controller.adjust_parameters(average_performance_metric)

        writer.add_scalar('Loss/Average', average_loss, epoch)
        writer.add_scalar('Performance/Average', average_performance_metric, epoch)

        print(f"Epoch {epoch+1}, Loss: {average_loss}, Performance Metric: {average_performance_metric}")

    writer.close()
    return network

