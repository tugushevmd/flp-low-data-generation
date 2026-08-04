# Low-data generation of FLP-like molecules

This repository studies how the chemical composition of a pretraining corpus affects molecular generation in a small and atypical target domain: frustrated Lewis pair (FLP)-like structures.

The central controlled experiment keeps the tokenizer, GRU architecture, corpus size, training budget and FLP fine-tuning data fixed. Only the fraction of main-group molecules in pretraining changes: `P0` (0%), `P0.1` (0.1%), `P1` (1%) and `P5` (5%). MolGPT, REINVENT and GP-MoLFormer are included as external reference models, not as a causal architecture comparison.

![External generator benchmark](figures/figure_4_external_generators.png)

## Main results

- The frozen-prior FLP BPC gap decreases from `1.134` for P0 to `0.530` for P5.
- With 166 FLP training molecules, P5 improves over P0 in all six paired training seeds.
- At 166 molecules, the mean P5-P0 changes are `+5.36` percentage points in validity, `+4.29` points in strict FLP-like yield and `+2.78` points in final candidate yield.
- The corresponding one-sided paired sign tests give `p = 0.0156`.
- At 83 molecules the direction is generally favourable, but generation-level effects are less stable.
- The final-candidate yields of the external models at 166 molecules are 14.5% for GP-MoLFormer, 20.7% for MolGPT and 29.6% for REINVENT.
- An independent blinded chemical review accepted 68 of 72 sampled final candidates (94.4%, Wilson 95% CI 86.6-97.8%).

These results support a limited claim: a chemically closer prior improves low-data adaptation in the controlled GRU experiment. `Strict FLP-like` is a structural screening rule. It does not establish frustration, catalytic activity or synthetic accessibility.

## Repository structure

```text
data/          Curated FLP split and controlled pretraining corpus
notebooks/     GPU experiments in execution order
evaluation/    Frozen FLP evaluator v2.1.1
scripts/       Recalculation, baselines and publication figures
results/       Aggregated results and compact raw controlled runs
figures/       Publication figures in PNG/PDF/SVG
tests/         Evaluator and dataset checks
configs/       Frozen experiment metadata
artifacts/     Instructions for large weights and external raw runs
```

## Quick start

Create the analysis environment:

```bash
conda env create -f environment.yml
conda activate flp-low-data-generation
```

Run all checks:

```bash
python -m unittest discover -s tests -v
```

Rebuild the final figures and tables:

```bash
python scripts/build_publication_figures_v211.py
```

The command reads the compact tables in `results/` and writes the figures to `figures/`.

## Recalculate the controlled experiment

The raw generation archives needed for this step are included because they are small:

```bash
python scripts/evaluate_controlled_priors_v2.py \
  --archive results/raw_controlled/pretraining/controlled_prior_pretraining_results.zip \
  --out-dir results/recomputed/controlled_priors

python scripts/evaluate_controlled_prior_learning_curves_v2.py \
  --archive results/raw_controlled/learning_curves/controlled_prior_learning_curves_results.zip \
  --out-dir results/recomputed/controlled_prior_learning_curves

python scripts/evaluate_controlled_prior_confirmatory_v2.py \
  --archive results/raw_controlled/confirmatory/controlled_prior_confirmatory_results.zip \
  --out-dir results/recomputed/controlled_prior_confirmatory \
  --learning-dir results/recomputed/controlled_prior_learning_curves
```

## Repeat GPU training

Run the notebooks in order:

1. `01_controlled_prior_pretraining.ipynb`
2. `02_controlled_prior_learning_curves.ipynb`
3. `03_controlled_prior_confirmatory.ipynb`

Upload `data/controlled_priors/controlled_prior_corpora.zip` to Colab or Kaggle before the first run. Each notebook saves its result and weight archives for the next stage. The three external-model notebooks reproduce the reference-model track.

PyTorch should be installed for the CUDA version available on the machine. Colab and Kaggle already provide it; avoid replacing their PyTorch build unless necessary.

## Reproducibility notes

- Training seeds: `11, 22, 33`; confirmatory seeds: `44, 55, 66`.
- Generation seeds: `101, 202, 303`.
- Generation seeds are averaged within each training run. The training seed is the unit of replication.
- The evaluator was frozen at version `2.1.1` before the final recalculation.
- The manual validation sample was drawn after the automatic filter was frozen.
- Model weights and large external archives are documented in [artifacts/README.md](artifacts/README.md) and are not stored in Git.

Further details are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [data/README.md](data/README.md) and [evaluation/README.md](evaluation/README.md).

