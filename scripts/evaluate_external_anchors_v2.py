import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import pandas as pd
from rdkit import rdBase


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import EVALUATOR_VERSION, audit_generation


parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["gpmolformer", "molgpt", "reinvent"])
parser.add_argument("archive")
parser.add_argument("--out-dir")
args = parser.parse_args()

archive_path = Path(args.archive).resolve()
out_dir = (
    ROOT / args.out_dir
    if args.out_dir
    else ROOT / "results" / "recomputed" / f"external_{args.model}"
)
out_dir.mkdir(parents=True, exist_ok=True)

data_dir = ROOT / "data" / "flp_curation"
curation_dir = data_dir


def read_smiles(path, column="smiles"):
    return pd.read_csv(path)[column].dropna().astype(str).tolist()


train_order = (
    pd.read_csv(
        ROOT / "data" / "controlled_priors" / "learning_curve_subsets.csv"
    )
    .sort_values("order")["smiles"]
    .astype(str)
    .tolist()
)
reference_smiles = read_smiles(
    curation_dir / "reference_smiles_curated_v2.csv",
    "canonical_smiles",
)
template_smiles = read_smiles(
        data_dir / "template_candidates.csv",
    "canonical_smiles",
)

run_rows = []
candidate_tables = []

with ZipFile(archive_path) as archive:
    sample_files = sorted(
        name
        for name in archive.namelist()
        if re.fullmatch(r"fraction_(25|100)_train_\d+_generation_\d+\.csv", name)
    )

    for name in sample_files:
        match = re.search(r"fraction_(\d+)_train_(\d+)_generation_(\d+)", name)
        fraction, training_seed, generation_seed = map(int, match.groups())
        samples = pd.read_csv(archive.open(name))

        smiles_column = "smiles" if "smiles" in samples else "SMILES"
        reached_limit = (
            samples["reached_max_length"].fillna(False).astype(bool).tolist()
            if "reached_max_length" in samples
            else [False] * len(samples)
        )
        train_smiles = train_order[:42 if fraction == 25 else 166]

        audit, summary, funnel = audit_generation(
            raw_smiles=samples[smiles_column].fillna("").astype(str).tolist(),
            train_smiles=train_smiles,
            seed_smiles=reference_smiles,
            template_smiles=template_smiles,
            reached_max_length=reached_limit,
        )
        audit["fraction"] = fraction
        audit["training_seed"] = training_seed
        audit["generation_seed"] = generation_seed
        funnel["fraction"] = fraction
        funnel["training_seed"] = training_seed
        funnel["generation_seed"] = generation_seed

        stem = f"fraction_{fraction}_train_{training_seed}_generation_{generation_seed}"
        audit.to_csv(out_dir / f"{stem}_evaluated.csv", index=False)
        funnel.to_csv(out_dir / f"{stem}_funnel.csv", index=False)

        row = summary.iloc[0].to_dict()
        row.update({
            "model": args.model,
            "fraction": fraction,
            "training_seed": training_seed,
            "generation_seed": generation_seed,
        })
        run_rows.append(row)

        candidates = audit[audit["is_final_candidate"]].copy()
        candidates["fraction"] = fraction
        candidates["training_seed"] = training_seed
        candidates["generation_seed"] = generation_seed
        candidate_tables.append(candidates)

        print(
            fraction,
            training_seed,
            generation_seed,
            "| validity", round(row["validity"], 3),
            "| strict", round(row["strict_flp_yield"], 3),
            "| final", round(row["final_candidate_yield"], 3),
        )

runs = pd.DataFrame(run_rows)
runs.to_csv(out_dir / "seed_matrix_summary_v2.csv", index=False)

metrics = [
    "validity",
    "unique_valid_yield",
    "chemically_sane_fraction",
    "neutral_fraction",
    "strict_flp_yield",
    "novel_flp_yield",
    "final_candidate_yield",
    "near_duplicate_095",
    "mean_snn_to_train",
    "internal_diversity",
    "scaffold_novelty_vs_train",
]
training_means = (
    runs.groupby(["fraction", "training_seed"])[metrics]
    .mean()
    .reset_index()
)
training_means.to_csv(out_dir / "training_seed_summary_v2.csv", index=False)

aggregate = (
    training_means.groupby("fraction")[metrics]
    .agg(["mean", "std"])
    .stack(0, future_stack=True)
    .reset_index(names=["fraction", "metric"])
)
aggregate.to_csv(out_dir / "fraction_aggregate_v2.csv", index=False)

if candidate_tables:
    candidates = pd.concat(candidate_tables, ignore_index=True)
    counts = (
        candidates.groupby(["fraction", "canonical_smiles"])
        .agg(
            candidate_instances=("canonical_smiles", "size"),
            training_seed_count=("training_seed", "nunique"),
        )
        .reset_index()
    )
    candidate_union = (
        candidates.drop_duplicates(["fraction", "canonical_smiles"])
        .merge(counts, on=["fraction", "canonical_smiles"])
    )
else:
    candidate_union = pd.DataFrame()
candidate_union.to_csv(out_dir / "candidate_union_v2.csv", index=False)

manifest = {
    "model": args.model,
    "evaluator_version": EVALUATOR_VERSION,
    "rdkit_version": rdBase.rdkitVersion,
    "archive": archive_path.name,
    "archive_sha256": sha256(archive_path.read_bytes()).hexdigest(),
    "sample_files": len(sample_files),
    "fractions": sorted(runs["fraction"].unique().astype(int).tolist()),
    "training_seeds": [11, 22, 33],
    "generation_seeds": [101, 202, 303],
}
(out_dir / "evaluator_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print()
print(aggregate.round(4).to_string(index=False))
