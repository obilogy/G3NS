import torch
import torch.nn as nn
import pandas as pd

class C_Elegans_Network(nn.Module):
    def __init__(self, file_path, state_vector_length):
        super(C_Elegans_Network, self).__init__()
        self.adjacency_matrix, self.neuron_to_idx, self.neuron_clusters = self.create_adjacency_matrix_and_clusters(file_path)
        self.state_vector_length = state_vector_length
        num_neurons = len(self.neuron_to_idx)

        self.neuron_states = nn.ParameterList([nn.Parameter(torch.randn(self.state_vector_length), requires_grad=True) for _ in range(num_neurons)])
        self.gating_functions = nn.ModuleDict({
            'Sensory core': nn.Identity(),
            'Hubs': nn.Sequential(nn.Linear(self.state_vector_length, 20), nn.ReLU(), nn.Linear(20, self.state_vector_length)),
            'Periphery': nn.Sequential(nn.Linear(self.state_vector_length, 15), nn.ReLU(), nn.Linear(15, self.state_vector_length)),
            'Motor core': nn.Identity(),
            'default': nn.Linear(self.state_vector_length, self.state_vector_length)
        })

    def create_adjacency_matrix_and_clusters(self, file_path):
        edge_list_df = pd.read_csv(file_path)
        neuron_names = pd.concat([edge_list_df['Source'], edge_list_df['Target']]).unique()
        neuron_to_idx = {name: idx for idx, name in enumerate(neuron_names)}
        num_neurons = len(neuron_to_idx)
        adjacency_matrix = torch.zeros((num_neurons, num_neurons))
        neuron_clusters = {}

        for _, row in edge_list_df.iterrows():
            src_idx = neuron_to_idx[row['Source']]
            dest_idx = neuron_to_idx[row['Target']]
            weight = row['Weight']
            adjacency_matrix[src_idx, dest_idx] = weight
            neuron_clusters[row['Source']] = row['Cluster_Source']
            neuron_clusters[row['Target']] = row['Cluster_Target']

        self.sensory_neurons = edge_list_df[edge_list_df['Cluster_Source'] == 'Sensory core']['Source'].unique()
        self.motor_neurons = edge_list_df[edge_list_df['Cluster_Target'] == 'Motor core']['Target'].unique()

        return adjacency_matrix, neuron_to_idx, neuron_clusters

    def F_v(self, current_state, incoming_messages):
        updated_state = torch.tanh(current_state + torch.exp(-incoming_messages))
        return updated_state

    def forward(self, timesteps, external_inputs=None):
        device = self.neuron_states[0].device
        if external_inputs is not None:
            self.set_external_inputs(external_inputs, device)
        for _ in range(timesteps):
            self.update_states(device)

    def set_external_inputs(self, external_inputs, device):
        for neuron_name, input_value in external_inputs.items():
            if neuron_name in self.neuron_to_idx:
                idx = self.neuron_to_idx[neuron_name]
                self.neuron_states[idx].data = torch.tensor([input_value] * self.state_vector_length, dtype=torch.float32).to(device)

    def update_states(self, device):
        adjacency_matrix = self.adjacency_matrix.to(device)
        for i in range(len(self.neuron_to_idx)):
            neuron_name = list(self.neuron_to_idx.keys())[list(self.neuron_to_idx.values()).index(i)]
            cluster = self.neuron_clusters.get(neuron_name, 'default')
            current_state = self.neuron_states[i].to(device)
            incoming_messages = torch.matmul(adjacency_matrix[i], torch.stack([s.to(device) for s in self.neuron_states]))
            updated_state = self.F_v(current_state, incoming_messages)
            gated_state = self.gating_functions[cluster](updated_state)
            self.neuron_states[i].data = gated_state.data

    def get_motor_neuron_outputs(self):
        outputs = {}
        for neuron, cluster in self.neuron_clusters.items():
            if cluster == 'Motor core' and neuron in self.neuron_to_idx:
                idx = self.neuron_to_idx[neuron]
                outputs[neuron] = self.neuron_states[idx].data
        return outputs

    def define_possible_actions(self):
        functions = ['tanh', 'relu', 'sigmoid', 'identity']
        operations = ['+', '-', '*', '/']
        additional_ops = ['exp', 'exp_neg', 'log', 'sqrt', 'square']

        actions = []
        for func in functions:
            for op in operations:
                for add_op in additional_ops:
                    action = {'function': func, 'operation': op, 'additional': add_op}
                    actions.append(action)
        return actions

    def modify_F_v(self, action):
        torch_functions = {
            'tanh': torch.tanh,
            'relu': torch.relu,
            'sigmoid': torch.sigmoid,
            'identity': lambda x: x,
            'exp': torch.exp,
            'exp_neg': lambda x: torch.exp(-x),
            'log': torch.log,
            'sqrt': torch.sqrt,
            'square': lambda x: x ** 2
        }

        func = torch_functions[action['function']]
        add_func = torch_functions[action['additional']]
        op = action['operation']

        def F_v(current_state, incoming_messages):
            if op == '+':
                updated_state = func(current_state + add_func(incoming_messages))
            elif op == '-':
                updated_state = func(current_state - add_func(incoming_messages))
            elif op == '*':
                updated_state = func(current_state * add_func(incoming_messages))
            elif op == '/':
                updated_state = func(current_state / add_func(incoming_messages))
            return updated_state

        self.F_v = F_v

    def reset_F_v(self):
        # Reset the F_v function to its original or baseline state
        def F_v(current_state, incoming_messages):
            # Original or baseline implementation of F_v
            updated_state = torch.tanh(current_state + torch.exp(-incoming_messages))
            return updated_state

        self.F_v = F_v