# Quantum Computing Based Design of Multivariate Porous Materials (QC-MTV)


## About This Project
<img src="images/Figure1.jpg">
This work introduces a novel quantum computing algorithm designed to efficiently explore the vast potential space of MTV porous materials. We utilized qubits to represent the reticular architecture of these materials and incorporated compositional, structural, and balance constraints into the Hamiltonian. This approach identifies optimal arrangements of building blocks that meet all specified design criteria. 

## Usage

This repository includes **Jupyter notebook examples** and **Python scripts** to demonstrate the application of our Hamiltonian model to user-specified multivariate material configurations.

- `hamiltonina_model_test.ipynb` : Provides an example of running **SamplingVQE simulation** using a classical simulator. 
- `cost/` : Contains the implementation of **the Hamiltonian model.**
  - Includes the `MTVcost()` class, which formulates the cost function for material optimization.
- `visualize/` : Contains visualization utilities.
  - Includes the `draw_graph()` function for **network visualization** of material structures.

***

## Getting Started

NOTE: This package is primarily tested on Linux systems. We recommend using Linux for installation.

To use the provided tools, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/shinyoung3/QC-MTV.git
   cd QC-MTV
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter notebook for step-by-step usage:

   ```bash
   jupyter notebook hamiltonian_model_test.ipynb
   ```

***



## Citation

If you want to use our QC-MTV framework in your research, please cite:

>  Quantum Computing Based Design of Multivariate Porous Materials [[link]](https://arxiv.org/abs/2502.06339)





