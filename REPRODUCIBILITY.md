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

Of the 299 curated structures, 285 (95.3%) contain B. The other 14 are Al-containing systems, including two that also contain Si. Their inclusion reflects the historical breadth of the collected FLP literature; the primary endpoint remains B-centered.

The controlled priors contain 150,000 unique SMILES each. P0, P0.1, P1 and P5 contain 0, 150, 1,500 and 7,500 main-group structures, respectively. Replacement molecules were matched by molecular weight, heavy-atom count, ring counts, rotatable bonds and SMILES length.

Within the 7,500 selected main-group molecules, B occurs in 3,007 structures (40.1%), P in 3,030 (40.4%), N in 3,741 (49.9%), Si in 844 (11.3%), Al in 375 (5.0%) and Ge in 375 (5.0%). The selection targets were 3,000 B, 3,000 P, 750 Si, 375 Al and 375 Ge structures. A total of 935 molecules (12.5%) contain B together with P or N.

The composition audit additionally compares full-corpus SMILES length, token count, branch count, heavy atoms, ring counts, rotatable bonds, token frequencies and scaffold diversity. The maximum absolute SMD in the matched replacement sets is 0.112; after dilution into the full corpora, the maximum is 0.010. P0-to-P5 token Jensen-Shannon divergence is 0.00066.

The pretraining source pool was filtered against all 299 curated reference structures, including train and validation molecules. There are no exact matches to the curated or template reference sets. The maximum ECFP4 Tanimoto similarity to a curated reference is 0.396, below the fixed 0.4 limit. Split-file hashes and overlap checks are reported in `table_s9_protocol_audit.csv` and `table_s10_file_integrity.csv`.

## Model selection

Checkpoints are selected using validation BPC. Test-generation metrics and manual review decisions do not participate in checkpoint selection. The selected checkpoint varies across seeds. A fixed-8000-exposure sensitivity analysis gives nearly the same pooled P5-P0 BPC improvement as validation-based selection (`0.0193` versus `0.0187`).

Generation was also repeated from checkpoints fixed in advance at 8,000 exposures. Strict FLP-like yield was 8.22% for P0 and 12.36% for P5, giving a paired difference of +4.13 percentage points (95% CI +2.29 to +5.97; P5 higher in 6/6 seeds). Validation-selected checkpoints gave +4.29 points. Relative to those selected checkpoints, fixing training at 8,000 exposures changed strict yield by 0.14 points for P0 and 0.28 points for P5, much less than the 4.13-4.29 point P5-P0 effect. Final-candidate yield at the fixed checkpoint was 4.28% for P0 and 6.73% for P5, a difference of +2.46 points (95% CI +1.01 to +3.90; 6/6 paired wins).

The frozen-prior FLP BPC gap is the difference between FLP and general-domain BPC before adaptation. The post-fine-tuning value above is P0 minus P5 raw FLP validation BPC after adaptation. Fine-tuning largely closes the likelihood difference between the priors, but their generation yields remain different; the two quantities describe different stages of the experiment.

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

For pooled P5 runs at the 166-molecule training size, mean yield is 12.64% at the strict FLP-like stage, 10.93% after reference-set novelty filtering and 7.07% after the final similarity and candidate filters. Thus, 55.9% of strict FLP-like generations remain final candidates.

## Statistical analysis

Strict FLP-like yield is the primary generation metric; validity and final-candidate yield are secondary descriptive metrics. P5 and P0 are compared within the same training seeds.

Seeds 11, 22 and 33 form the discovery cohort. Seeds 44, 55 and 66 form the independent confirmatory cohort. Results are reported separately for both cohorts, with the pooled six-seed analysis used only as supporting evidence. Mean paired effects are accompanied by t-based 95% confidence intervals, exact sign-flip tests and sign tests. The sign test reaches its minimum possible one-sided value of 0.125 (`1/8`) in each three-seed cohort and 0.0156 (`1/64`) after pooling. Effect sizes and confidence intervals carry the quantitative interpretation.

The four-level frozen-prior dose response is tested with an exact blocked Page-style permutation test over all 13,824 within-seed label permutations. Its `p = 7.23e-5` is the minimum attainable value (`1/13,824`). Fine-tuning trends are analysed separately for each FLP training-set size. At the 166-molecule size, the three-level P0 < P1 < P5 strict-yield result also reaches its permutation floor, `p = 0.00463` (`1/216`).

P0.1 was evaluated at the frozen-prior stage but was not included in FLP fine-tuning. The generation-level dose analysis therefore contains P0, P1 and P5 only.

## Manual chemical validation

The external-model audit used a fresh stratified sample of 72 final candidates: 12 structures per model and training fraction. The reviewer was shown structure identifiers without model labels. Four structures were rejected. This first audit had one reviewer and no negative controls.

A second panel mixes 24 model candidates with 21 chemically valid, filter-derived decoys. The structures were randomly ordered and labelled only with blind identifiers. All 24 model candidates were accepted; 13 of 21 decoys were rejected. Three accepted decoys were Al/Si-only systems outside the B-centered study scope. Their acceptance was chemically reasonable and is consistent with a reviewer applying broader FLP chemistry rather than mechanically reproducing the B-centered endpoint. Within the primary scope, 13 of 18 decoys were rejected. These are operational panel metrics because filter-derived decoys are not an absolute chemical ground truth. A second reviewer would still be needed to estimate inter-rater agreement.

## Baselines and external models

The character 5-gram model is the data-matched non-neural baseline trained on the same 166 FLP molecules. The fragment recombination library is a rule-based enumerator and is therefore treated as a chemical upper bound rather than a data-matched learner. It gives a 50.3% fragment-candidate yield after strict, curated-seed novelty and similarity filtering. Template-relative final novelty is zero by construction because the enumerated library is also included in the template reference.

Library proximity is therefore measured continuously rather than by exact novelty. Among 586 unique P5 final candidates, median nearest-library ECFP4 Tanimoto similarity is 0.426; 1.54% reach 0.80, 0.51% reach 0.90, none reach 0.95, and 15.5% share a Murcko scaffold with a library member. The corresponding P0 values are similar (median 0.415, 0.54% at or above 0.80 and 13.2% scaffold overlap). P5 increases the number of final candidates without materially changing their proximity to the enumerated fragment space. The effect is therefore quantitative rather than a shift into a distinct fragment-defined region.

Controlled GRU P5, MolGPT, REINVENT and GP-MoLFormer are evaluated with the same frozen filter, but their architectures and pretraining corpora differ. These cross-model results provide context and do not isolate a causal architecture effect. The archived GP-MoLFormer benchmark contains only the 42-molecule experiment. An earlier draft quoted a 14.5% final-candidate yield from an exploratory 166-molecule run, but its raw generation archive was not retained and the result could not be re-evaluated under evaluator 2.1.1. It is therefore omitted from the frozen comparison.

## Software freeze

- Evaluator: `2.1.1`
- Python: `3.11`
- RDKit and analysis dependencies: see `environment.yml`
- Model checkpoints and revisions: see `configs/model_manifest.yaml`

The code is released under the MIT License. Source datasets and pretrained model checkpoints remain subject to their original licenses and terms.
