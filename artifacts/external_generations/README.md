# External generation archives

These compact archives contain the raw generated SMILES and run settings used for
the external-model comparisons. TensorBoard event logs and model weights were
omitted because they are not required to recalculate the reported molecular metrics.

- `gpmolformer_42_raw.zip`: archived GP-MoLFormer 42-molecule experiment.
- `molgpt_raw.zip`: MolGPT generations and run settings.
- `reinvent_raw.zip`: REINVENT generations and run settings.
- `manifest.json`: source-archive hashes and included-file lists.

The complete GP-MoLFormer 166-molecule generation denominator was not retained and
is not represented as a quantitative benchmark. A small retained candidate subset is
used only for qualitative structural illustration.
