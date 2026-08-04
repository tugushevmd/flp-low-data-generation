# Data

## `flp_curation/`

This directory contains the manually curated FLP reference and the frozen train/validation split used throughout the study.

- `reference_smiles_curated_v2.csv`: 299 retained reference structures.
- `reference_smiles_excluded_v2.csv`: structures excluded during manual curation.
- `train_smiles.csv`: 166 training molecules.
- `validation_smiles.csv`: 20 validation molecules.
- `template_candidates.csv`: chemically motivated template library used only as a novelty reference and baseline.
- `split_manifest.json`: counts and SHA-256 checksums.

## `controlled_priors/`

`controlled_prior_corpora.zip` contains four matched pretraining corpora with 150,000 unique SMILES each:

| corpus | main-group fraction | main-group molecules |
|---|---:|---:|
| P0 | 0% | 0 |
| P0.1 | 0.1% | 150 |
| P1 | 1% | 1,500 |
| P5 | 5% | 7,500 |

The main-group pool contains B, P, Si, Al and Ge structures. P0 molecules replaced by main-group examples were descriptor-matched to keep molecular size and simple topology approximately constant. The maximum absolute standardized mean difference is 0.112.

The corpus was assembled from a standardized ZINC250k source and PubChem PUG-REST records. Exact matches to the curated FLP and template references were excluded, and the maximum nearest-neighbour similarity to the curated FLP reference was restricted to less than 0.4.

The source-level checksums and construction parameters are recorded in `corpus_manifest.json`. Raw ZINC and PubChem downloads are not duplicated in this repository.

