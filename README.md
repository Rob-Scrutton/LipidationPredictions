# Lipidation Predictions

Predicting structural outcomes of lipidated elastin-like peptides (ELPs) using ESM2 protein language model embeddings.

## Overview

This repository contains code and data for classifying lipidated ELP variants into three structural classes: **droplet**, **fiber**, or **metastable**. We use embeddings from ESM2 (`esm2_t33_650M_UR50D`) as input features for Random Forest classifiers.

## Dataset

- **76 peptide variants** with variable amino acids at positions 2-4 (lipidation sites)
- Sequence format: `G[XXX]` where XXX represents the three variable residues (A, G, L, S, V, W)
- Conserved ELP backbone: `(GVGVP)₃₁-GWP`

## Repository Structure

```
├── extract_esm2_embeddings.py   # ESM2 embedding extraction script
├── lipidation_classifier.ipynb  # Classification and analysis notebook
└── data/
    ├── all_sequences.fasta      # Input sequences
    ├── sequence_labels.csv      # Structural class labels
    ├── embeddings_whole_seq/    # Mean-pooled embeddings (.pt files)
    └── embeddings_lipid_sites/  # Lipid site-specific embeddings (.pt files)
```

## Usage

### Extract embeddings
```bash
python extract_esm2_embeddings.py \
    --fasta_file data/all_sequences.fasta \
    --output_dir data/embeddings/ \
    --lipid_positions 2 3 4
```

### Run classification
Open `lipidation_classifier.ipynb` to reproduce the analysis, including:
- UMAP visualizations
- Random Forest classification with 5-fold stratified CV
- ROC curves with bootstrap confidence intervals
- SHAP interpretability analysis

## Requirements

- Python 3.8+
- PyTorch
- fair-esm
- scikit-learn
- umap-learn
- shap

## Citations

**ESM2:**
```bibtex
@article{
doi:10.1126/science.ade2574,
author = {Zeming Lin  and Halil Akin  and Roshan Rao  and Brian Hie  and Zhongkai Zhu  and Wenting Lu  and Nikita Smetanin  and Robert Verkuil  and Ori Kabeli  and Yaniv Shmueli  and Allan dos Santos Costa  and Maryam Fazel-Zarandi  and Tom Sercu  and Salvatore Candido  and Alexander Rives },
title = {Evolutionary-scale prediction of atomic-level protein structure with a language model},
journal = {Science},
volume = {379},
number = {6637},
pages = {1123-1130},
year = {2023},
doi = {10.1126/science.ade2574},
URL = {https://www.science.org/doi/abs/10.1126/science.ade2574},
eprint = {https://www.science.org/doi/pdf/10.1126/science.ade2574},
```

<!-- Add your paper citation here when published -->
<!-- **This work:**
```bibtex
@article{...}
```
-->
