# Reproducibility protocol

## Study scope

The primary endpoint is restricted to boron-centered FLP-like molecules. A qualifying Lewis-acid centre is neutral tricoordinate boron; qualifying Lewis-base centres are the phosphorus and nitrogen motifs defined by evaluator 2.1.1. Al-, Si- and Ge-only Lewis-acid systems are outside the endpoint rather than chemically invalid. This boundary is fixed for all model comparisons.

## Experimental unit

Each model is trained independently for every training seed. Three generation seeds are sampled from each trained checkpoint and averaged before comparisons. Generation seeds are therefore repeated measurements, not independent replicates.

The primary controlled learning curves contain three training seeds (`11`, `22`, `33`). The confirmatory P5 versus P0 experiment adds three new seeds (`44`, `55`, `66`) and reports paired results across all six seeds.

## Data split

- Curated FLP reference: 299 structures.
- Training split: 166 structures.
- Validation split: 20 structures.
- Learning-curve subsets: 42, 83 and 166 structures.
- Subsets are nested and selected before SMILES augmentation.

The controlled priors contain 150,000 unique SMILES each. P0, P0.1, P1 and P5 contain 0, 150, 1,500 and 7,500 main-group structures, respectively. Replacement molecules were matched by molecular weight, heavy-atom count, ring counts, rotatable bonds and SMILES length.

The composition audit additionally compares full-corpus SMILES length, token count, branch count, heavy atoms, ring counts, rotatable bonds, token frequencies and scaffold diversity. The maximum absolute SMD in the matched replacement sets is 0.112; after dilution into the full corpora, the maximum is 0.010. P0-to-P5 token Jensen-Shannon divergence is 0.00066.

The pretraining source pool was filtered against all 299 curated reference structures, including train and validation molecules. There are no exact matches to the curated or template reference sets. The maximum ECFP4 Tanimoto similarity to a curated reference is 0.396, below the fixed 0.4 limit. Split-file hashes and overlap checks are reported in `table_s9_protocol_audit.csv` and `table_s10_file_integrity.csv`.

## Model selection

Checkpoints are selected using validation BPC. Test-generation metrics and manual review decisions do not participate in checkpoint selection. The selected checkpoint varies across seeds. A fixed-8000-exposure sensitivity analysis gives nearly the same pooled P5-P0 BPC improvement as validation-based selection (`0.0193` versus `0.0187`).

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

Strict FLP-like yield is the primary generation metric; validity and final-candidate yield are secondary descriptive metrics. P5 and P0 are compared within the same training seeds.

Seeds 11, 22 and 33 form the discovery cohort. Seeds 44, 55 and 66 form the independent confirmatory cohort. Results are reported separately for both cohorts, with the pooled six-seed analysis used only as supporting evidence. Mean paired effects are accompanied by t-based 95% confidence intervals, exact sign-flip tests and sign tests. The sign test reaches its minimum possible one-sided value of 0.125 in each three-seed cohort and 0.0156 after pooling.

The four-level frozen-prior dose response is tested with an exact blocked Page-style permutation test over all 13,824 within-seed label permutations. Fine-tuning trends are analysed separately for each FLP training-set size.

## Manual chemical validation

The external-model audit used a fresh stratified sample of 72 final candidates: 12 structures per model and training fraction. The reviewer was shown structure identifiers without model labels. Four structures were rejected. This first audit had one reviewer and no negative controls.

A second panel mixes 24 model candidates with 21 chemically valid, filter-derived decoys. The structures were randomly ordered and labelled only with blind identifiers. All 24 model candidates were accepted; 13 of 21 decoys were rejected. Three accepted decoys were Al/Si-only systems outside the final B-centered scope. Restricted to the 18 within-scope decoys, 13 were rejected. These are operational panel metrics because filter-derived decoys are not an absolute chemical ground truth. A second reviewer would still be needed to estimate inter-rater agreement.

## Baselines and external models

The character 5-gram model is the data-matched non-neural baseline trained on the same 166 FLP molecules. The fragment recombination library is a rule-based enumerator and is therefore treated as a chemical upper bound rather than a data-matched learner.

Controlled GRU P5, MolGPT, REINVENT and GP-MoLFormer are evaluated with the same frozen filter, but their architectures and pretraining corpora differ. These cross-model results provide context and do not isolate a causal architecture effect. GP-MoLFormer has a 42-molecule run only; no 166-molecule value is reported.

## Software freeze

- Evaluator: `2.1.1`
- Python: `3.11`
- RDKit and analysis dependencies: see `environment.yml`
- Model checkpoints and revisions: see `configs/model_manifest.yaml`

The code is released under the MIT License. Source datasets and pretrained model checkpoints remain subject to their original licenses and terms.
