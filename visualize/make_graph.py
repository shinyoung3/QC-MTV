import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib.patches as mpatches

def draw_graph(G, colors=None, pos=None, figsize=(6, 4), output_path=None):
    """
    Draws a graph with edge colors determined by the order of weights using the colormap.
    Optionally saves the figure to a file if output_path is provided.

    Args:
        G (networkx.Graph): The graph to be drawn.
        pos (dict, optional): Node positions as a dictionary {node: (x, y)}.
        figsize (tuple, optional): Figure size for the plot.
        output_path (str, optional): Path to save the figure (optional).
    """
    if pos is None:
        pos = nx.spring_layout(G)  # Default layout if not provided

    fig, ax = plt.subplots(figsize=figsize)  # Set user-defined figure size

    # Set node colors (default: gray) **
    if colors is None:
        colors = ['gray' for _ in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=600, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)

    # Assign edge colors based on unique weight values
    edge_weights = nx.get_edge_attributes(G, 'weight')
    unique_weights = sorted(set(edge_weights.values()))  # Get unique weights
    num_weights = len(unique_weights)

    # Define a colormap for unique edge weights
    cmap = plt.get_cmap('coolwarm', num_weights)
    weight_to_color = {weight: cmap(i / max(1, num_weights - 1)) for i, weight in enumerate(unique_weights)}

    # Draw edges with respective colors
    for (u, v, weight) in G.edges(data='weight'):
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            edge_color=[weight_to_color[weight]],
            width=2,
            ax=ax
        )

    # Create legend manually for edge colors
    legend_patches = [mpatches.Patch(color=weight_to_color[w], label=f"{w}") for w in unique_weights]
    ax.legend(handles=legend_patches, title="Edge Weights", loc="upper right", fontsize=10)

    # Save the figure if output_path is provided
    if output_path:
        plt.savefig(output_path, format="jpg", dpi=300, bbox_inches="tight")
        print(f"Graph saved successfully to {output_path}")
        plt.close(fig)
    else:
        plt.show()
