import argparse
import io
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import EVALUATOR_VERSION, audit_generation


parser = argparse.ArgumentParser()
parser.add_argument(
    "--archive",
    default="controlled_prior_learning_curves_results.zip",
)
parser.add_argument(
    "--out-dir",
    default="results/recomputed/controlled_prior_learning_curves",
)
args = parser.parse_args()

archive_path = Path(args.archive)
if not archive_path.is_absolute():
    archive_path = ROOT / archive_path

out_dir = ROOT / args.out_dir
out_dir.mkdir(parents=True, exist_ok=True)

curation_dir = ROOT / "data" / "flp_curation"
subset_table = pd.read_csv(
    ROOT
    / "data"
    / "controlled_priors"
    / "learning_curve_subsets.csv"
)
validation_smiles = (
    pd.read_csv(curation_dir / "validation_smiles.csv")["smiles"]
    .dropna()
    .astype(str)
    .tolist()
)
seed_smiles = (
    pd.read_csv(
        curation_dir / "reference_smiles_curated_v2.csv"
    )["canonical_smiles"]
    .dropna()
    .astype(str)
    .tolist()
)
template_smiles = (
    pd.read_csv(
        ROOT
        / "results"
        / "baselines"
        / "fragment_recombination"
        / "fragment_library.csv"
    )["canonical_smiles"]
    .dropna()
    .astype(str)
    .tolist()
)

fraction_sizes = {25: 42, 50: 83, 100: 166}
pattern = re.compile(
    r"learning_curve_(P0|P1|P5|scratch)"
    r"_fraction_(25|50|100)"
    r"_train_(\d+)_generation_(\d+)\.csv"
)

run_rows = []
candidate_tables = []

with ZipFile(archive_path) as archive:
    sample_names = sorted(
        name
        for name in archive.namelist()
        if pattern.fullmatch(name)
    )

    for name in sample_names:
        match = pattern.fullmatch(name)
        family = match.group(1)
        fraction = int(match.group(2))
        training_seed = int(match.group(3))
        generation_seed = int(match.group(4))

        train_smiles = (
            subset_table.iloc[:fraction_sizes[fraction]]["smiles"]
            .astype(str)
            .tolist()
        )
        samples = pd.read_csv(io.BytesIO(archive.read(name)))
        reached_limit = (
            samples["reached_max_length"]
            .fillna(False)
            .astype(bool)
            .tolist()
        )

        audit, summary, _ = audit_generation(
            raw_smiles=samples["generated_smiles"].fillna("").tolist(),
            train_smiles=train_smiles,
            seed_smiles=seed_smiles,
            template_smiles=template_smiles,
            reached_max_length=reached_limit,
        )

        audit["family"] = family
        audit["fraction"] = fraction
        audit["training_seed"] = training_seed
        audit["generation_seed"] = generation_seed
        audit.to_csv(
            out_dir
            / (
                f"{family}_{fraction}_train_{training_seed}"
                f"_generation_{generation_seed}_evaluated.csv"
            ),
            index=False,
        )

        row = summary.iloc[0].to_dict()
        row.update({
            "family": family,
            "fraction": fraction,
            "training_seed": training_seed,
            "generation_seed": generation_seed,
        })
        run_rows.append(row)
        candidate_tables.append(
            audit[audit["is_final_candidate"]].copy()
        )

        print(
            family,
            "|", str(fraction) + "%",
            "| train", training_seed,
            "| generation", generation_seed,
            "| validity", round(row["validity"], 3),
            "| strict FLP", round(row["strict_flp_yield"], 3),
        )

if len(run_rows) != 108:
    raise ValueError(
        f"Found {len(run_rows)} runs; expected 108"
    )

run_table = pd.DataFrame(run_rows)
run_table.to_csv(out_dir / "run_summary_v2.csv", index=False)

metric_columns = [
    column
    for column in run_table.columns
    if column not in {
        "evaluator_version",
        "family",
        "fraction",
        "training_seed",
        "generation_seed",
    }
]

training_seed_summary = (
    run_table
    .groupby(["family", "fraction", "training_seed"])[metric_columns]
    .mean()
    .reset_index()
)
training_seed_summary.to_csv(
    out_dir / "training_seed_summary_v2.csv",
    index=False,
)

aggregate_rows = []
for (family, fraction), part in training_seed_summary.groupby(
    ["family", "fraction"]
):
    row = {
        "family": family,
        "fraction": fraction,
        "training_seeds": len(part),
    }
    for metric in metric_columns:
        row[f"{metric}_mean"] = part[metric].mean()
        row[f"{metric}_sd"] = part[metric].std(ddof=1)
    aggregate_rows.append(row)

aggregate = pd.DataFrame(aggregate_rows)
aggregate.to_csv(out_dir / "aggregate_v2.csv", index=False)

paired_metrics = [
    "validity",
    "strict_flp_yield",
    "final_candidate_yield",
]
paired_rows = []
for fraction in [25, 50, 100]:
    base = (
        training_seed_summary[
            (training_seed_summary["family"] == "P0")
            & (training_seed_summary["fraction"] == fraction)
        ]
        .set_index("training_seed")
    )
    for family in ["P1", "P5", "scratch"]:
        part = (
            training_seed_summary[
                (training_seed_summary["family"] == family)
                & (training_seed_summary["fraction"] == fraction)
            ]
            .set_index("training_seed")
        )
        for training_seed in base.index:
            row = {
                "family": family,
                "fraction": fraction,
                "training_seed": training_seed,
            }
            for metric in paired_metrics:
                row[f"{metric}_delta_vs_P0"] = (
                    part.loc[training_seed, metric]
                    - base.loc[training_seed, metric]
                )
            paired_rows.append(row)

pd.DataFrame(paired_rows).to_csv(
    out_dir / "paired_deltas_vs_P0.csv",
    index=False,
)

candidate_union = pd.concat(candidate_tables, ignore_index=True)
candidate_union = (
    candidate_union
    .sort_values("snn_to_train", ascending=False)
    .drop_duplicates("canonical_smiles")
)
candidate_union.to_csv(
    out_dir / "candidate_union_v2.csv",
    index=False,
)

with ZipFile(archive_path) as archive:
    best_runs = pd.read_csv(io.BytesIO(archive.read("best_runs.csv")))
best_runs.to_csv(out_dir / "best_runs.csv", index=False)

bpc_summary = (
    best_runs
    .groupby(["family", "fraction"])["best_validation_bpc"]
    .agg(["mean", "std"])
    .reset_index()
)
bpc_summary.to_csv(
    out_dir / "validation_bpc_summary.csv",
    index=False,
)

colors = {
    "P0": "#686D70",
    "P1": "#7663A5",
    "P5": "#963F52",
    "scratch": "#356859",
}
panels = [
    ("unique_valid_yield", "Unique-valid yield"),
    ("strict_flp_yield", "Strict FLP-like yield"),
    ("final_candidate_yield", "Final-candidate yield"),
]

fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))
for (metric, title), ax in zip(panels, axes.flat[:3]):
    for family in colors:
        part = (
            aggregate[aggregate["family"] == family]
            .sort_values("fraction")
        )
        ax.errorbar(
            part["fraction"],
            part[f"{metric}_mean"],
            yerr=part[f"{metric}_sd"],
            marker="o",
            capsize=4,
            color=colors[family],
            label=family,
        )
    ax.set_title(title)
    ax.set_xlabel("FLP training fraction, %")
    ax.set_xticks([25, 50, 100])

ax = axes.flat[3]
for family in colors:
    part = (
        bpc_summary[bpc_summary["family"] == family]
        .sort_values("fraction")
    )
    ax.errorbar(
        part["fraction"],
        part["mean"],
        yerr=part["std"],
        marker="o",
        capsize=4,
        color=colors[family],
        label=family,
    )
ax.set_title("Validation BPC")
ax.set_xlabel("FLP training fraction, %")
ax.set_xticks([25, 50, 100])

for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)
axes.flat[0].legend()

fig.tight_layout()
fig.savefig(
    out_dir / "controlled_prior_learning_curves_v2.png",
    dpi=180,
    bbox_inches="tight",
)

print()
print("Evaluator:", EVALUATOR_VERSION)
print(
    aggregate[[
        "family",
        "fraction",
        "validity_mean",
        "unique_valid_yield_mean",
        "strict_flp_yield_mean",
        "final_candidate_yield_mean",
    ]].round(4).to_string(index=False)
)
print("Unique final candidates:", len(candidate_union))
