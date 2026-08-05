# FLP evaluator 2.1.1

The evaluator provides a reproducible structural screen for generated SMILES in the boron-centered FLP-like domain.

It checks:

1. SMILES parsing and RDKit sanitization;
2. single-component, neutral structures without radicals or unsupported atoms;
3. a neutral tricoordinate boron Lewis-acid centre;
4. an allowed phosphine or nitrogen Lewis-base centre;
5. absence of a direct B-P or B-N bond;
6. novelty relative to curated and template references;
7. similarity to the FLP training domain.

Phosphorus directly bonded to oxygen is not classified as a phosphine Lewis base. Version 2.1.1 also distinguishes chemically acceptable tricoordinate B-B motifs from undercoordinated or hydrido B-B forms, which remain blocked.

`Strict FLP-like` is a graph-based screening label. It is not proof of frustrated Lewis-pair behaviour, reactivity, catalytic performance, stability or synthetic accessibility.

Al-, Si- and Ge-only Lewis-acid systems are outside the evaluator scope. Their exclusion should be read as a study boundary, not as a claim that these motifs cannot form Lewis pairs.

The control structures are stored in `controls.csv`, and all evaluator checks can be run with:

```bash
python -m unittest tests.test_evaluator -v
```
