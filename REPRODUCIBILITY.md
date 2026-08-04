# Reproducibility protocol

## Experimental unit

Each model is trained independently for every training seed. Three generation seeds are sampled from each trained checkpoint and averaged before comparisons. Generation seeds are therefore repeated measurements, not independent replicates.

The primary controlled learning curves contain three training seeds (`11`, `22`, `33`). The confirmatory P5 versus P0 experiment adds three new seeds (`44`, `55`, `66`) and reports paired results across all six seeds.

## Data split

- Curated FLP reference: 299 structures.
- Training split: 166 structures.
- Validation split: 20 structures.
- Learning-curve subsets: 42, 83 and 166 structures.
- Subsets are nested and selected before SMILES augmentation.

The controlled priors contain 150,000 unique SMILES each. P0, P0.1, P1 and P5 differ only in the fraction of main-group structures. Replacement molecules were matched by simple molecular descriptors, and exact or near FLP-reference leakage was excluded during corpus construction.

## Model selection

Checkpoints are selected using validation BPC. Test-generation metrics and manual review decisions do not participate in checkpoint selection.

## Reported metrics

- validity;
- unique-valid yield;
- strict FLP-like yield;
- novel FLP-like yield;
- final-candidate yield;
- similarity to the training set;
- internal diversity;
- scaffold novelty.

The final-candidate filter requires a valid, chemically sane, neutral, single-component strict FLP-like structure, novelty relative to the reference sets, and a predefined similarity window to the training domain.

## Statistical analysis

P5 and P0 are compared within the same training seeds. With six paired observations, the confirmatory analysis uses an exact one-sided sign test and reports the individual paired changes. The test is deliberately modest: it establishes consistency across the tested seeds, not a universal ranking of pretraining strategies.

## Manual chemical validation

The external-model audit used a fresh stratified sample of 72 final candidates: 12 structures per model and training fraction. The reviewer was shown structure identifiers without model labels. Four structures were rejected. The automatic filter was not changed after this confirmatory review.

## Software freeze

- Evaluator: `2.1.1`
- Python: `3.11`
- RDKit and analysis dependencies: see `environment.yml`
- Model checkpoints and revisions: see `configs/model_manifest.yaml`

