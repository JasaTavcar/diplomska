# Diploma Thesis Experiments: FreeViz and Supervised PCA

This repository contains reproducible experiments for a diploma thesis focused on dimensionality reduction with:

- **FreeViz** (classification-oriented visualization/projection)
- **Supervised PCA (SPCA)** via manifold optimization (regression-oriented projection)

The repository is organized so that methods are implemented as a small reusable library, data is stored in a dedicated `data/` directory, and experiments are stored in `experiments/`.

## Methods Implemented

- `experiments/models/freeviz_torch.py`
  - PyTorch implementation of FreeViz optimization
  - Includes projection, training loop, and kNN prediction/evaluation helpers
- `experiments/models/spca_manifold.py`
  - SPCA implementation on the Grassmann manifold
  - Includes manifold gradient, Armijo line search, geodesic updates, and normalized losses
- `experiments/utils.py`
  - Shared data loading and utility metrics:
    - variation explained
    - prediction error
    - feature scaling helpers

## Data

- `data/zoo.tab`: zoological classification dataset used for FreeViz experiments
- `data/parkinsons.data`: regression dataset used for SPCA and SPCA-vs-PCA comparison
- `data/car.data`: car evaluation dataset used in FreeViz vs SPCA comparison
- [Mushroom](https://archive.ics.uci.edu/dataset/73/mushroom) (UCI ID 73): downloaded via `ucimlrepo` for FreeViz vs SPCA comparison
- [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (UCI ID 17): downloaded via `ucimlrepo` for FreeViz vs SPCA comparison

## Environment and Dependencies

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies (listed in `requirements.txt`)

```bash
pip install -r requirements.txt
```

## How to Run Experiments

All the experiments should be completely reproducible

Run all commands from the repository root.

All the experiment notebooks are saved in `/experiments`. They should be run by executing all the cells from top to bottom in order.

## Experiment Overview and Expected Outputs

### 1) `freeviz_visualize_zoo.ipynb`

- **Goal:** Example usage of FreeViz on concrete zoo data
- **Method:** Train FreeViz projection and visualize class separation in 2D
- **Outputs:** 2D projection plot(s), learned anchor interpretation

### 2) `freeviz_classification_eval.ipynb`

- **Goal:** Basic quantitative evaluation for FreeViz
- **Method:** 10-fold cross-validation with kNN in projected space
- **Outputs:** classification performance metrics (fold-wise and aggregate)

### 3) `spca.ipynb`

- **Goal:** SPCA experiment on regression data
- **Method:** Train SPCA across multiple `lambda` values (trade-off between prediction and reconstruction)
- **Outputs:** metric curves/plots (prediction vs reconstruction behavior), selected projection visualization

### 4) `spca_vs_pca.ipynb`

- **Goal:** Clear method comparison: **SPCA vs regular PCA**
- **Method:** Side-by-side projection and loading comparison on the same sampled data
- **Outputs:** comparative plots and tables (embeddings, loadings, and metric summaries)

### 5) `spca_bair.ipynb`

- **Goal:** SPCA experiment using Bair variable ordering/importance
- **Method:** Apply SPCA with Bair-informed variable selection/ranking
- **Outputs:** projection visualizations and metric summaries with Bair-based ordering

### 6) `spca_bair_vs_manifold.ipynb`

- **Goal:** Comparison of **SPCA with Bair ordering vs manifold-optimized SPCA**
- **Method:** Side-by-side comparison of the two SPCA approaches on the same data
- **Outputs:** comparative plots and tables showing differences between Bair-based and manifold-optimized projections

### 7) `spca_classification_zoo.ipynb`

- **Goal:** Supervised PCA with classification on the zoological dataset
- **Method:** SPCA Classification model jointly learns a 2D latent projection and a linear classifier, balancing reconstruction and classification via $\lambda$ regularization
- **Outputs:** 2D latent projection plots, classification decision boundaries, reconstruction loss and classification accuracy curves across $\lambda$ values

### 8) `compare_freeviz_spca.ipynb`

- **Goal:** Compare resulting visualizations of FreeViz and SPCA for classification
- **Method:** Train both FreeViz and SPCA classification models on zoo, mushroom, and diagnostic breast cancer datasets, then plot their 2D projections side by side
- **Outputs:** side-by-side 2D projection plots of FreeViz vs SPCA for each dataset


## Basic Result Evaluation Included So Far

The repository includes baseline evaluation suitable for generating thesis artifacts such as plots and tables.:

- **Classification evaluation** for FreeViz (`freeviz_classification_eval.ipynb`)
- **Regression-oriented objective evaluation** for SPCA (`spca.ipynb`)
- **Direct comparative analysis** of SPCA and PCA (`spca_vs_pca.ipynb`)
- **SPCA with Bair ordering** experiments (`spca_bair.ipynb`)
- **Comparison of Bair vs manifold SPCA** (`spca_bair_vs_manifold.ipynb`)
- **FreeViz vs SPCA visualization comparison** (`compare_freeviz_spca.ipynb`)
- **SPCA classification on zoo data** (`spca_classification_zoo.ipynb`)

## Notes for Thesis Integration

- Current experiments are notebook-based; if needed, notebooks can be exported to standalone scripts for one-command CI execution.

