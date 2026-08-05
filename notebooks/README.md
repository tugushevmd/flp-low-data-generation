# Notebooks

The controlled experiment is run in this order:

1. `01_controlled_prior_pretraining.ipynb`
2. `02_controlled_prior_learning_curves.ipynb`
3. `03_controlled_prior_confirmatory.ipynb`

The remaining notebooks reproduce the independent external-model benchmarks:

- `04_external_gpmolformer.ipynb`
- `05_external_molgpt.ipynb`
- `06_external_reinvent.ipynb`

The notebooks are written for Colab or Kaggle GPU sessions. They save result and weight ZIP archives at the end of each run. Use the PyTorch build supplied by the platform instead of replacing it with a CUDA-specific wheel. Do not commit generated weights to Git; their expected names and checksums are listed in `artifacts/README.md`.
