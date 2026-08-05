# Results

- `publication_tables/`: compact tables used in the manuscript figures.
- `controlled_priors/`: frozen-prior BPC and zero-shot summaries.
- `controlled_prior_learning_curves/`: three-seed controlled learning curves.
- `controlled_prior_confirmatory/`: six-seed paired P5 versus P0 analysis.
- `external_*`: compact external-model summaries.
- `manual_validation/`: decisions and Wilson intervals from the blinded audit.
- `blind_decoy_review/`: the scored candidate-plus-decoy review and its image pages.
- `raw_controlled/`: small raw generation archives needed to rerun the evaluator.

`publication_tables/` contains five main tables and eleven supplementary tables. The corresponding six main figures and four supplementary figures are stored in `figures/`. Run `python scripts/rebuild_release.py` to rebuild the complete analysis set and execute the tests.

Large model weights and the 37 MB REINVENT raw archive are intentionally excluded. See `artifacts/README.md`.
