# AntsNet

**An R Package Demonstrating the Isomorphism Between Ant Colony Generational Learning and Neural Networks**

> Companion code for *"Isomorphic Functionalities between Ant Colony and Ensemble Learning: Part III — Gradient Descent, Neural Plasticity, and the Emergence of Deep Intelligence"* by Ernest Fokoué, Gregory Babbitt, and Yuval Levental (Rochester Institute of Technology).

---

## Overview

AntsNet implements both **Generational Ant Colony Learning (GACL)** and a **multi-layer perceptron trained with SGD**, and provides simulation tools to demonstrate that these two systems are mathematically isomorphic:

| Neural Network Concept | Ant Colony Analog |
|---|---|
| Training epoch *t* | Generation *g* |
| Network weights **w** | Pheromone configuration **τ** |
| Mini-batch | Recruitment wave |
| Forward pass | Ant foraging guided by pheromone |
| Loss function *L*(**w**) | Negative colony fitness −*F*(**τ**) |
| Learning rate *η* | Evaporation rate *ρ* |
| Backpropagation | Credit assignment via recruitment intensity |
| Weight update **w** ← **w** − *η*∇*L* | Pheromone update **τ** ← (1−*ρ*)**τ** + *γ*∇*F* |

## Quick Start

### Prerequisites

No external dependencies — AntsNet uses only base R.

### Generate All Manuscript Figures

From the repository root:

```
Rscript generate_figures.R
```

This produces `figures/` containing:

| File | Manuscript Reference | Description |
|---|---|---|
| `figure1_gradient_isomorphism.pdf` | Figure 1 (Section 4.2) | Normalized error/loss trajectories |
| `figure2_learning_curves.pdf` | Figure 2 (Section 5.2) | 20-replicate learning curves with SE bands |
| `figure3_pheromone_weight_evolution.pdf` | Figure 3 (Section 5.1) | Pheromone ↔ weight evolution |
| `figure4_learning_rate_sensitivity.pdf` | Figure 4 (Section 5.2) | Optimal ρ ≅ optimal η |
| `figure5_noise_robustness.pdf` | Figure 5 (Section 5.3) | Identical degradation under noise |
| `figure6_convergence_complexity.pdf` | Figure 6 (Section 5.2) | Convergence across 3 complexity levels |
| `figure7_gradient_dynamics.pdf` | Figure 7 (Section 5.1) | Error signal and gradient magnitude |
| `figure8_plasticity_adaptation.pdf` | Figure 8 (Section 5.4) | Adaptation to environmental shift |
| `figure9_uniform_convergence.pdf` | Figure 9 (Section 5.2) | Var ~ N⁻¹·⁴⁴ convergence rate |
| `figure9b_trajectory_convergence.pdf` | Figure 9b (Section 5.2) | Trajectory fan-in at N=10, 100, 1000 |

### Use as a Package

```r
# Install from local source
install.packages(".", repos = NULL, type = "source")

library(AntsNet)

# Run GACL
result <- gacl(c(10, 7, 5, 4, 3), n_generations = 50)
plot(result$fitness_history, type = "l")

# Run neural network
d <- generate_synthetic_data(n = 500, p = 5, complexity = 2)
nn <- simple_neural_network(d$X[1:400, ], d$y[1:400], n_epochs = 50)
plot(nn$val_acc, type = "l")

# Reproduce all manuscript figures
plot_isomorphism()                # Figure 1
plot_learning_curves()            # Figure 2
plot_pheromone_weight()           # Figure 3
plot_learning_rate_sensitivity()  # Figure 4
plot_noise_robustness()           # Figure 5
plot_convergence_complexity()     # Figure 6
plot_gradient_dynamics()          # Figure 7
plot_plasticity()                 # Figure 8
```

## Package Structure

```
IsomorphismSim_Part_III/
├── R/
│   ├── algorithms.R        # GACL + neural network + data generation
│   └── plotting.R          # All plotting functions (Figures 1–8)
├── figures/                 # Pre-generated manuscript figures (PDF)
├── man/                     # R documentation (.Rd files)
├── generate_figures.R       # Standalone runner (no install needed)
├── isomorphism_ant_colony_neural_networks.tex   # Manuscript source
├── biological_ant_neural.bib                    # Bibliography
├── DESCRIPTION
├── NAMESPACE
├── LICENSE
└── README.md
```

## Key Functions

| Function | Purpose |
|---|---|
| `gacl(site_qualities, ...)` | Generational Ant Colony Learning |
| `simple_neural_network(X, y, ...)` | MLP trained with mini-batch SGD |
| `generate_synthetic_data(n, p, ...)` | Synthetic classification data |
| `plot_isomorphism()` | Figure 1: Gradient descent isomorphism |
| `plot_learning_curves()` | Figure 2: Learning curves comparison |
| `plot_pheromone_weight()` | Figure 3: Pheromone vs weight evolution |
| `plot_learning_rate_sensitivity()` | Figure 4: Rate sensitivity |
| `plot_noise_robustness()` | Figure 5: Noise robustness |
| `plot_convergence_complexity()` | Figure 6: Convergence across complexity |
| `plot_gradient_dynamics()` | Figure 7: Error signal and gradient |
| `plot_plasticity()` | Figure 8: Adaptation to environmental change |

## Citation

```
@article{fokoue2026neural,
  title   = {Isomorphic Functionalities between Ant Colony and Ensemble Learning:
             Part {III} --- Gradient Descent, Neural Plasticity, and the
             Emergence of Deep Intelligence},
  author  = {Fokou{\'e}, Ernest and Babbitt, Gregory and Levental, Yuval},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT
