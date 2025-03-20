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
    """
    A class that defines the Hamiltonian cost function for multivariate (MTV) porous material design.

    This class encodes the structural, compositional, and balance constraints of the system into 
    a quadratic optimization problem, which can be solved using quantum algorithms.

    Args:
        G (nx.Graph): A graph representation where nodes represent linker sites.
        required_counts (dict): Dictionary specifying the required number of each linker type in the final configuration.
        linkers (list): A list of available linker types.
        lengths (dict):  A dictionary mapping each linker type to its characteristic length.
        E (float, optional): A scaling factor for the balance constraint in the cost function. Default is 1.
    """
    def __init__(self, G: nx.Graph, required_counts, linkers, lengths, E: float = 1):
        self.G = G
        self.required_counts = required_counts
        self.linkers = linkers
        self.lengths = lengths
        self.E = E
        
    def to_quadratic_program(self) -> QuadraticProgram:
        """
        Constructs the quadratic optimization problem for MTV material design.

        This function formulates a Quadratic Program (QP) by defining constraints for 
        linker composition, balance, and occupancy. The objective function is built 
        based on minimizing the deviation from required linker counts, ensuring 
        structural balance, and enforcing occupancy constraints.

        Returns:
            QuadraticProgram: The formulated quadratic optimization problem ready for
                              quantum or classical solvers.
        """
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
        """
        Interprets the binary bitstring from OptimizationResult into a human-readable linker assignment.

        This function decodes the bitstring into linker configurations based on 
        user-specified linker types and site constraints.

        Args:
            result (OptimizationResult): The output from a quantum or classical optimizer.

        Returns:
            List[str]: A list representing the linker assignments for each site.
        """
        num_sites = self.G.number_of_nodes()  
        num_linkers = len(self.linkers)  
        interpreted_result = []

        bitstring = ''.join(map(str, result.x.astype(int)))

        for i in range(num_sites):
            start_idx = i * num_linkers
            end_idx = start_idx + num_linkers
            site_encoding = bitstring[start_idx:end_idx]

            assigned_linkers = [self.linkers[j] for j, bit in enumerate(site_encoding) if bit == '1']
            interpreted_result.append(",".join(assigned_linkers) if assigned_linkers else "-")  
        return interpreted_result
    
    def _draw_result(self, result: OptimizationResult, pos: Optional[Dict[int, np.ndarray]] = None) -> None:
        """
        Draws the resulting graph with nodes colored based on linker assignments and adds a legend.

        This visualization helps to understand the spatial distribution of linkers in the 
        optimized material design.

        Args:
            result (OptimizationResult): The output from a quantum or classical optimizer.
            pos (dict, optional): The positions of nodes in the graph layout.
        """
        solution = self.interpret(result)  
        color_map = self._get_linker_color_map()  
        colors = self._node_color(solution, color_map)  

        plt.figure(figsize=(9, 5))
        nx.draw(self.G, pos=pos, node_color=colors, with_labels=True, edge_color="gray", node_size=1000)
        plt.title("MTV Linker Configuration")

        unique_linkers = sorted(set(l for node in solution for l in node.split(",") if l in color_map))
        legend_patches = [mpatches.Patch(color=color_map[l], label=l) for l in unique_linkers]

        plt.legend(handles=legend_patches, title="Linker Types", loc="upper right", fontsize=10)
        plt.show()

    def _get_linker_color_map(self):
        """
        Returns a predefined color map for linkers.

        Each linker type is assigned a unique color for visualization purposes.
        """
        return {"A": "r", "B": "g", "C": "b", "D": "c", "-": "gray"} 

    def _node_color(self, solution: List[str], color_map: Dict[str, str]) -> List[str]:
        """
        Assigns colors based on linker presence for visualization.

        Args:
            solution (List[str]): List of linkers assigned to each node.
            color_map (Dict[str, str]): Predefined mapping of linkers to colors.

        Returns:
            List[str]: List of colors corresponding to each node.
        """
        node_colors = []
        for node in solution:
            assigned_linkers = node.split(",") if node != "-" else []
            if assigned_linkers:
                node_colors.append(color_map.get(assigned_linkers[0], "gray"))
            else:
                node_colors.append("gray")  
        return node_colors

    def plot_distribution(self, samples):
        """
        Plots a histogram of linker configurations based on bitstring probabilities.

        This function visualizes the probability distribution of different material 
        configurations obtained from quantum or classical optimization.

        Args:
            samples: The sampled bitstrings and their corresponding probabilities.
        """
        configurations = []
        probabilities = []

        for sample in samples:
            bitstring = ''.join(map(str, sample.x.astype(int)))  
            interpreted = self.interpret(sample)  
            configurations.append(str(interpreted))  
            probabilities.append(sample.probability) 

        sorted_indices = sorted(range(len(probabilities)), key=lambda k: probabilities[k], reverse=True)
        configurations = [configurations[i] for i in sorted_indices]
        probabilities = [probabilities[i] for i in sorted_indices]

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(configurations)), probabilities, tick_label=configurations)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.xlabel("Linker Configurations")
        plt.ylabel("Probability")
        plt.title("Probability Distribution of Linker Configurations")
        plt.show()
