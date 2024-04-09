import networkx as nx
import pandas as pd
from pyvis.network import Network

def visualize_network_interactively(file_path, output_file='network_graph.html'):
    # Read the edge list from a CSV file
    edge_list_df = pd.read_csv(file_path)

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph
    for _, row in edge_list_df.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'])

    # Create a Pyvis network from the NetworkX graph
    net = Network(notebook=False)
    net.from_nx(G)

    # Set options (can be customized further)
    net.set_options("""
    var options = {
      "nodes": {
        "scaling": {
          "min": 10,
          "max": 30
        }
      },
      "edges": {
        "color": {
          "inherit": true
        },
        "smooth": false
      },
      "physics": {
        "barnesHut": {
          "centralGravity": 0.1,
          "springLength": 100,
          "springConstant": 0.02
        }
      }
    }
    """)

    # Save and open the network in browser
    net.show(output_file)

if __name__ == "__main__":
    file_path = 'output_with_clusters.csv'  # Update with your actual file path
    visualize_network_interactively(file_path)
