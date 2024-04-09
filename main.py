import torch.optim as optim
from network import C_Elegans_Network
from environment import SimulatedEnvironment
from controller import NetworkController
from ppo import PPO, PPOPolicyNetwork
from utils import get_state_representation, train_with_environment

def main():
    file_path = 'output_with_clusters.csv'  # Update with your actual file path
    state_vector_length = 10  # Update as per your requirement

    network = C_Elegans_Network(file_path, state_vector_length)
    ppo_policy = PPOPolicyNetwork(num_inputs=80, num_actions=20)  # Update these parameters as needed
    optimizer = optim.RMSprop(ppo_policy.parameters(), lr=0.001)
    ppo_agent = PPO(policy_network=ppo_policy, optimizer=optimizer)

    trained_network = train_with_environment(network, ppo_agent, epochs=100, num_samples=50)

if __name__ == "__main__":
    main()
