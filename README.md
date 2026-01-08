# 🦠 Numerical Epidemic Simulation: SVEIR Model with Spatial Diffusion

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-NumPy%20%7C%20SciPy%20%7C%20Matplotlib-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Contributions](https://img.shields.io/badge/Contributions-welcome-brightgreen.svg)

<p align="center">
  <em>A numerical simulation of epidemic spread including population movement (diffusion).</em>
</p>

<!-- Optional: Add a GIF of your simulation here for greater impact! -->
<!-- 
<p align="center">
  <img src="path/to/your/simulation_animation.gif" alt="SVEIR Diffusion Simulation" width="700"/>
</p>
-->

## 📌 Project Overview
This project presents a numerical simulation of an epidemic, specifically **Influenza**, using an extended **SVEIR (Susceptible-Vaccinated-Exposed-Infected-Recovered)** compartmental model.

A key feature of this simulation is the inclusion of **spatial diffusion**, which models the movement of populations across different geographical regions. This transforms the standard system of Ordinary Differential Equations (ODEs) into a more complex system of **Partial Differential Equations (PDEs)**. The simulation solves this system numerically to provide a realistic depiction of how an epidemic spreads not just over time, but also across space.

## ✨ Key Features
*   **SVEIR Compartmental Model:** Tracks five distinct population groups: Susceptible, Vaccinated, Exposed, Infected, and Recovered.
*   **Spatial Diffusion:** Implements a 1D diffusion term using the Laplacian operator to simulate the geographical spread of the epidemic.
*   **Vaccination Scenarios:** Allows for the analysis of different vaccination rates and their impact on flattening the infection curve.
*   **Numerical Solvers:**
    *   **Spatial Discretization:** Uses the **Finite Difference Method** to discretize the spatial domain.
    *   **Temporal Integration:** Employs robust time-stepping algorithms from `scipy.integrate.solve_ivp` to solve the resulting system of ODEs.
*   **Data Generation & Visualization:** Includes scripts to generate simulation data for various parameters and notebooks for in-depth analysis and visualization.


## 📂 Repository Structure
```
.
├── data/               # Generated datasets for different infection parameters (IP1, IP2)
├── foundations/        # Foundational numerical algorithms (Root finding, Integration)
├── notebooks/          # Jupyter notebooks for analysis and visualization
│   └── Epidemic_Analysis.ipynb
├── src/                # Core Python scripts for the simulation logic
└── README.md
```

## 🚀 Getting Started

### Prerequisites
*   Python 3.8 or higher
*   Jupyter Notebook or JupyterLab

### Installation
1.  **Clone the repository**
2.  **Create a virtual environment (recommended)**
3.  **Install dependencies**
### Running the Simulation
1.  The main analysis and visualization is performed in the Jupyter notebook.
2.  Launch the Jupyter environment:
3.  Navigate to the `notebooks/` directory and open `Epidemic_Analysis.ipynb`.
4.  Run the cells in the notebook to execute the simulation and visualize the results.

## 🛠️ Tech Stack
*   **Core Language:** Python 3
*   **Numerical Computation:** NumPy, SciPy (`solve_ivp`)
*   **Data Visualization:** Matplotlib, Seaborn
*   **Notebook Environment:** Jupyter
