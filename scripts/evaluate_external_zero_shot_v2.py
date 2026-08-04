import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pandas as pd
from rdkit import rdBase


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import EVALUATOR_VERSION, audit_generation


parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default="results/recomputed/external_zero_shot")
args = parser.parse_args()

raw_dir = ROOT / "results" / "external_anchors" / "raw"
out_dir = ROOT / args.out_dir
out_dir.mkdir(parents=True, exist_ok=True)

archives = {
    "GP-MoLFormer": raw_dir / "external_gpmolformer_25_results.zip",
    "MolGPT": raw_dir / "external_molgpt_results.zip",
    "REINVENT": raw_dir / "external_reinvent_results.zip",
}

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
hashes = {}

for model, archive_path in archives.items():
    hashes[model] = sha256(archive_path.read_bytes()).hexdigest()
    with ZipFile(archive_path) as archive:
        for generation_seed in [101, 202, 303]:
            name = f"pretrained_generation_{generation_seed}.csv"
            samples = pd.read_csv(archive.open(name))
            smiles_column = "smiles" if "smiles" in samples else "SMILES"
            reached_limit = (
                samples["reached_max_length"].fillna(False).astype(bool).tolist()
                if "reached_max_length" in samples
                else [False] * len(samples)
            )

            for fraction, size in [(25, 42), (100, 166)]:
                audit, summary, funnel = audit_generation(
                    raw_smiles=samples[smiles_column].fillna("").astype(str).tolist(),
                    train_smiles=train_order[:size],
                    seed_smiles=reference_smiles,
                    template_smiles=template_smiles,
                    reached_max_length=reached_limit,
                )
                audit["model"] = model
                audit["fraction"] = fraction
                audit["generation_seed"] = generation_seed
                funnel["model"] = model
                funnel["fraction"] = fraction
                funnel["generation_seed"] = generation_seed

                stem = f"{model.lower().replace('-', '').replace(' ', '_')}_{fraction}_{generation_seed}"
                audit.to_csv(out_dir / f"{stem}_evaluated.csv", index=False)
                funnel.to_csv(out_dir / f"{stem}_funnel.csv", index=False)

                row = summary.iloc[0].to_dict()
                row.update({
                    "model": model,
                    "fraction": fraction,
                    "generation_seed": generation_seed,
                })
                run_rows.append(row)

                print(
                    model,
                    fraction,
                    generation_seed,
                    "| validity", round(row["validity"], 3),
                    "| strict", round(row["strict_flp_yield"], 3),
                    "| final", round(row["final_candidate_yield"], 3),
                )

runs = pd.DataFrame(run_rows)
runs.to_csv(out_dir / "zero_shot_seed_summary_v2.csv", index=False)

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
aggregate = (
    runs.groupby(["model", "fraction"])[metrics]
    .agg(["mean", "std"])
    .stack(0, future_stack=True)
    .reset_index(names=["model", "fraction", "metric"])
)
aggregate.to_csv(out_dir / "zero_shot_aggregate_v2.csv", index=False)

manifest = {
    "evaluator_version": EVALUATOR_VERSION,
    "rdkit_version": rdBase.rdkitVersion,
    "archives_sha256": hashes,
    "generation_seeds": [101, 202, 303],
    "fractions": {"25": 42, "100": 166},
}
(out_dir / "evaluator_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print()
print(aggregate.round(4).to_string(index=False))
