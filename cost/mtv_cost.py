import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from docplex.mp.model import Model
from collections import Counter 
from typing import Union, Dict, Optional, List
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.problems.quadratic_program import QuadraticProgram
from qiskit_optimization.applications.graph_optimization_application import GraphOptimizationApplication
       
        
class MTVcost(GraphOptimizationApplication):
    def __init__(self, G: nx.Graph, required_counts, linkers, lengths, E: float = 1):
        self.G = G
        self.required_counts = required_counts
        self.linkers = linkers
        self.lengths = lengths
        self.E = E
        
    def to_quadratic_program(self) -> QuadraticProgram:
        mdl = Model(name="MTV Porous Material Cost Function")

        q = {f'q_{i}_{t}': mdl.binary_var(name=f'q_{i}_{t}') for i in range(self.G.number_of_nodes()) for t in self.linkers}
        
        ratio_cost = 200 * mdl.sum((mdl.sum(q[f'q_{i}_{t}'] for i in range(self.G.number_of_nodes())) - self.required_counts[t])**2 for t in self.linkers)
        
        edge_values = {}
        balance_cost_terms = []

        for (i, j, d) in self.G.edges(data=True):
            edge_values[(i, j)] = mdl.sum(
                self.lengths[t1] * q[f'q_{i}_{t1}'] + self.lengths[t2] * q[f'q_{j}_{t2}']
                for t1 in self.linkers for t2 in self.linkers
            )

        mean_edge_value = mdl.sum(edge_values[(i, j)] for (i, j) in self.G.edges) / len(self.G.edges)

        for (i, j) in self.G.edges:
            balance_cost_terms.append(
                self.G.edges[i, j]["weight"] * (edge_values[(i, j)] - mean_edge_value) ** 2
            )
        
        balance_cost = mdl.sum(balance_cost_terms)

        occupancy_cost = 300 * mdl.sum((mdl.sum(q[f'q_{i}_{t}'] for t in self.linkers) - 1) ** 2 for i in range(self.G.number_of_nodes()))

        total_cost = ratio_cost + self.E * balance_cost + occupancy_cost
        mdl.minimize(total_cost)

        op = from_docplex_mp(mdl)
        return op

    def interpret(self, result: OptimizationResult) -> List[str]:
        """Interprets the binary bitstring from OptimizationResult into a human-readable linker assignment."""
        num_sites = self.G.number_of_nodes()  # Number of linker sites
        num_linkers = len(self.linkers)  # Number of linker types
        interpreted_result = []

        bitstring = ''.join(map(str, result.x.astype(int)))

        # Decode each linker site
        for i in range(num_sites):
            start_idx = i * num_linkers
            end_idx = start_idx + num_linkers
            site_encoding = bitstring[start_idx:end_idx]

            # Determine which linkers are assigned to this site
            assigned_linkers = [self.linkers[j] for j, bit in enumerate(site_encoding) if bit == '1']
            interpreted_result.append(",".join(assigned_linkers) if assigned_linkers else "-")  # '-' for empty sites

        return interpreted_result
    
    def _draw_result(self, result: OptimizationResult, pos: Optional[Dict[int, np.ndarray]] = None) -> None:
        """Draws the resulting graph with nodes colored based on linker assignments and adds a legend.

        Args:
        result: The OptimizationResult containing the best configuration.
        pos: The positions of nodes.
        """
        solution = self.interpret(result)  # Convert result to linker assignments
        color_map = self._get_linker_color_map()  # Get predefined color map
        colors = self._node_color(solution, color_map)  # Get node colors

        plt.figure(figsize=(9, 5))
        nx.draw(self.G, pos=pos, node_color=colors, with_labels=True, edge_color="gray", node_size=1000)
        plt.title("MTV Linker Configuration")

        # ** Create legend dynamically based on linkers present **
        unique_linkers = sorted(set(l for node in solution for l in node.split(",") if l in color_map))
        legend_patches = [mpatches.Patch(color=color_map[l], label=l) for l in unique_linkers]

        plt.legend(handles=legend_patches, title="Linker Types", loc="upper right", fontsize=10)
        plt.show()

    def _get_linker_color_map(self):
        """Returns a predefined color map for linkers."""
        return {"A": "r", "B": "g", "C": "b", "D": "c", "-": "gray"}  # "-" represents an empty node

    def _node_color(self, solution: List[str], color_map: Dict[str, str]) -> List[str]:
        """Assigns colors based on linker presence.

        Args:
            solution: List of linkers assigned to each node.
            color_map: Predefined mapping of linkers to colors.

        Returns:
            List of colors corresponding to each node.
        """
        node_colors = []
        for node in solution:
            assigned_linkers = node.split(",") if node != "-" else []
            if assigned_linkers:
                node_colors.append(color_map.get(assigned_linkers[0], "gray"))  # Use first linker for coloring
            else:
                node_colors.append("gray")  # Default to gray for empty nodes
        return node_colors

    def plot_distribution(self, samples):
        """Plots a histogram of linker configurations based on bitstring probabilities."""
        configurations = []
        probabilities = []

        # Process each sample
        for sample in samples:
            bitstring = ''.join(map(str, sample.x.astype(int)))  # Convert numpy array to string
            interpreted = self.interpret(sample)  # Convert bitstring to linker configuration
            configurations.append(str(interpreted))  # Store as string for x-axis labeling
            probabilities.append(sample.probability)  # Store probability for y-axis

        # Sort configurations based on probability for better visualization
        sorted_indices = sorted(range(len(probabilities)), key=lambda k: probabilities[k], reverse=True)
        configurations = [configurations[i] for i in sorted_indices]
        probabilities = [probabilities[i] for i in sorted_indices]

        # Plot the histogram
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(configurations)), probabilities, tick_label=configurations)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.xlabel("Linker Configurations")
        plt.ylabel("Probability")
        plt.title("Probability Distribution of Linker Configurations")
        plt.show()
