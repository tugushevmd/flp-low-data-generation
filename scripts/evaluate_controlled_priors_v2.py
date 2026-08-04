import argparse
import io
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import EVALUATOR_VERSION, audit_generation


parser = argparse.ArgumentParser()
parser.add_argument(
    "--archive",
    default=(
        "results/controlled_prior_pretraining/"
        "controlled_prior_pretraining_results.zip"
    ),
)
parser.add_argument(
    "--out-dir",
    default="results/recomputed/controlled_priors",
)
args = parser.parse_args()

archive_path = Path(args.archive)
if not archive_path.is_absolute():
    archive_path = ROOT / archive_path
out_dir = ROOT / args.out_dir
out_dir.mkdir(parents=True, exist_ok=True)

curation_dir = ROOT / "data" / "flp_curation"

train_smiles = (
    pd.read_csv(curation_dir / "train_smiles.csv")["smiles"]
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


def atom_symbols(smiles):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return set()
    return {atom.GetSymbol() for atom in mol.GetAtoms()}


def element_metrics(audit):
    unique = audit[audit["is_unique_valid"]].copy()
    symbols = unique["canonical_smiles"].map(atom_symbols)
    metrics = {}

    for element in ["B", "P", "N", "Si", "Al", "Ge"]:
        metrics[f"{element}_fraction_unique_valid"] = (
            symbols.map(lambda values: element in values).mean()
        )

    main_group = {"B", "P", "Si", "Al", "Ge"}
    metrics["main_group_fraction_unique_valid"] = symbols.map(
        lambda values: bool(values & main_group)
    ).mean()
    metrics["B_and_P_or_N_fraction_unique_valid"] = symbols.map(
        lambda values: (
            "B" in values
            and ("P" in values or "N" in values)
        )
    ).mean()
    return metrics


pattern = re.compile(
    r"zero_shot_(P0(?:\.1)?|P1|P5)"
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
        prior = match.group(1)
        training_seed = int(match.group(2))
        generation_seed = int(match.group(3))

        samples = pd.read_csv(
            io.BytesIO(archive.read(name))
        )
        reached_limit = (
            samples["reached_max_length"]
            .fillna(False)
            .astype(bool)
            .tolist()
        )

        audit, summary, funnel = audit_generation(
            raw_smiles=samples["generated_smiles"].fillna("").tolist(),
            train_smiles=train_smiles,
            seed_smiles=seed_smiles,
            template_smiles=template_smiles,
            reached_max_length=reached_limit,
        )

        audit["prior"] = prior
        audit["training_seed"] = training_seed
        audit["generation_seed"] = generation_seed
        audit.to_csv(
            out_dir
            / (
                f"{prior}_train_{training_seed}"
                f"_generation_{generation_seed}_evaluated.csv"
            ),
            index=False,
        )

        row = summary.iloc[0].to_dict()
        row.update(element_metrics(audit))
        row["prior"] = prior
        row["training_seed"] = training_seed
        row["generation_seed"] = generation_seed
        run_rows.append(row)

        candidates = audit[audit["is_final_candidate"]].copy()
        candidate_tables.append(candidates)

        print(
            prior,
            "| train", training_seed,
            "| generation", generation_seed,
            "| validity", round(row["validity"], 3),
            "| main-group", round(
                row["main_group_fraction_unique_valid"],
                3,
            ),
            "| strict FLP", round(row["strict_flp_yield"], 3),
        )

run_table = pd.DataFrame(run_rows)
run_table.to_csv(out_dir / "run_summary_v2.csv", index=False)

metric_columns = [
    column
    for column in run_table.columns
    if column not in {
        "evaluator_version",
        "prior",
        "training_seed",
        "generation_seed",
    }
]

training_seed_summary = (
    run_table
    .groupby(["prior", "training_seed"])[metric_columns]
    .mean()
    .reset_index()
)
training_seed_summary.to_csv(
    out_dir / "training_seed_summary_v2.csv",
    index=False,
)

aggregate_rows = []
for prior, part in training_seed_summary.groupby("prior"):
    row = {"prior": prior, "training_seeds": len(part)}
    for metric in metric_columns:
        row[f"{metric}_mean"] = part[metric].mean()
        row[f"{metric}_sd"] = part[metric].std(ddof=1)
    aggregate_rows.append(row)

aggregate = pd.DataFrame(aggregate_rows)
prior_order = pd.Categorical(
    aggregate["prior"],
    categories=["P0", "P0.1", "P1", "P5"],
    ordered=True,
)
aggregate = (
    aggregate.assign(prior_order=prior_order)
    .sort_values("prior_order")
    .drop(columns="prior_order")
)
aggregate.to_csv(out_dir / "aggregate_v2.csv", index=False)

if candidate_tables:
    candidate_union = pd.concat(candidate_tables, ignore_index=True)
    candidate_union = (
        candidate_union
        .sort_values("snn_to_train", ascending=False)
        .drop_duplicates("canonical_smiles")
    )
else:
    candidate_union = pd.DataFrame()
candidate_union.to_csv(out_dir / "candidate_union_v2.csv", index=False)

bpc_path = out_dir / "frozen_bpc_runs.csv"
with ZipFile(archive_path) as archive:
    bpc_runs = pd.read_csv(
        io.BytesIO(archive.read("frozen_bpc_runs.csv"))
    )
bpc_runs.to_csv(bpc_path, index=False)

bpc_summary = (
    bpc_runs.groupby("prior")
    .agg(
        flp_gap_mean=("flp_gap", "mean"),
        flp_gap_sd=("flp_gap", "std"),
    )
    .reindex(["P0", "P0.1", "P1", "P5"])
)

plot_table = aggregate.set_index("prior").reindex(
    ["P0", "P0.1", "P1", "P5"]
)

colors = ["#686D70", "#356859", "#7663A5", "#963F52"]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

axes[0].errorbar(
    range(4),
    bpc_summary["flp_gap_mean"],
    yerr=bpc_summary["flp_gap_sd"],
    marker="o",
    color="#963F52",
    capsize=4,
)
axes[0].set_xticks(range(4), bpc_summary.index)
axes[0].set_ylabel("Разница FLP BPC и general BPC")
axes[0].set_title("Разрыв до дообучения")

axes[1].bar(
    range(4),
    plot_table["main_group_fraction_unique_valid_mean"],
    yerr=plot_table["main_group_fraction_unique_valid_sd"],
    color=colors,
    capsize=4,
)
axes[1].set_xticks(range(4), plot_table.index)
axes[1].set_ylabel("Доля среди уникальных валидных")
axes[1].set_title("Main-group элементы в исходной генерации")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig(
    out_dir / "controlled_prior_zero_shot.png",
    dpi=180,
    bbox_inches="tight",
)

print()
print("Evaluator:", EVALUATOR_VERSION)
print(aggregate[[
    "prior",
    "validity_mean",
    "unique_valid_yield_mean",
    "main_group_fraction_unique_valid_mean",
    "strict_flp_yield_mean",
    "final_candidate_yield_mean",
]].round(4).to_string(index=False))
print("Уникальных финальных кандидатов:", len(candidate_union))
