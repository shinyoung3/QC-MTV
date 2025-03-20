import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib.lines as mlines

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
        pos = nx.spring_layout(G) 

    fig, ax = plt.subplots(figsize=figsize)  

    if colors is None:
        colors = ['gray' for _ in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=600, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)

    edge_weights = nx.get_edge_attributes(G, 'weight')
    unique_weights = sorted(set(edge_weights.values())) 
    num_weights = len(unique_weights)

    cmap = plt.get_cmap('coolwarm', num_weights)
    weight_to_color = {weight: cmap(i / max(1, num_weights - 1)) for i, weight in enumerate(unique_weights)}

    for (u, v, weight) in G.edges(data='weight'):
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            edge_color=[weight_to_color[weight]],
            width=2,
            ax=ax
        )
   
    legend_lines = [
        mlines.Line2D(
            [], [], color=weight_to_color[w], linewidth=2, label=f"{w}"
        ) for w in unique_weights
    ]
    ax.legend(handles=legend_lines, title="Edge Weights", loc="upper right", fontsize=10)
    
    if output_path:
        plt.savefig(output_path, format="jpg", dpi=300, bbox_inches="tight")
        print(f"Graph saved successfully to {output_path}")
        plt.close(fig)
    else:
        plt.show()
