import argparse
import io
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import EVALUATOR_VERSION, audit_generation


parser = argparse.ArgumentParser()
parser.add_argument(
    "--archive",
    default="results/controlled_prior_confirmatory/controlled_prior_confirmatory_results.zip",
)
parser.add_argument(
    "--out-dir",
    default="results/recomputed/controlled_prior_confirmatory",
)
parser.add_argument(
    "--learning-dir",
    default="results/recomputed/controlled_prior_learning_curves",
)
args = parser.parse_args()

ARCHIVE = ROOT / args.archive
OUT_DIR = ROOT / args.out_dir
LEARNING_DIR = ROOT / args.learning_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

curation_dir = ROOT / "data" / "flp_curation"
subset_table = pd.read_csv(
    ROOT / "data" / "controlled_priors" / "learning_curve_subsets.csv"
)
seed_smiles = (
    pd.read_csv(curation_dir / "reference_smiles_curated_v2.csv")["canonical_smiles"]
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

fraction_sizes = {50: 83, 100: 166}
pattern = re.compile(
    r"confirmatory_(P0|P5)_fraction_(50|100)"
    r"_train_(44|55|66)_generation_(101|202|303)\.csv"
)

run_rows = []
candidate_tables = []

with ZipFile(ARCHIVE) as archive:
    sample_names = sorted(name for name in archive.namelist() if pattern.fullmatch(name))

    for name in sample_names:
        match = pattern.fullmatch(name)
        family = match.group(1)
        fraction = int(match.group(2))
        training_seed = int(match.group(3))
        generation_seed = int(match.group(4))

        train_smiles = (
            subset_table.iloc[:fraction_sizes[fraction]]["smiles"].astype(str).tolist()
        )
        samples = pd.read_csv(io.BytesIO(archive.read(name)))
        reached_limit = samples["reached_max_length"].fillna(False).astype(bool).tolist()

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
            OUT_DIR
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
        candidate_tables.append(audit[audit["is_final_candidate"]].copy())

        print(
            family,
            str(fraction) + "%",
            "train", training_seed,
            "generation", generation_seed,
            "validity", round(row["validity"], 3),
            "final", round(row["final_candidate_yield"], 3),
        )

if len(run_rows) != 36:
    raise ValueError(f"Найдено {len(run_rows)} запусков вместо 36")

run_table = pd.DataFrame(run_rows)
run_table.to_csv(OUT_DIR / "run_summary_v2.csv", index=False)

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

new_seed_summary = (
    run_table.groupby(["family", "fraction", "training_seed"])[metric_columns]
    .mean()
    .reset_index()
)
new_seed_summary.to_csv(OUT_DIR / "new_seed_summary_v2.csv", index=False)

old_seed_summary = pd.read_csv(
    LEARNING_DIR / "training_seed_summary_v2.csv"
)
old_seed_summary = old_seed_summary[
    old_seed_summary["family"].isin(["P0", "P5"])
    & old_seed_summary["fraction"].isin([50, 100])
]

combined = pd.concat([old_seed_summary, new_seed_summary], ignore_index=True)
combined = combined.sort_values(["fraction", "family", "training_seed"])
combined.to_csv(OUT_DIR / "combined_seed_summary_v2.csv", index=False)

aggregate_rows = []
for (family, fraction), part in combined.groupby(["family", "fraction"]):
    row = {"family": family, "fraction": fraction, "training_seeds": len(part)}
    for metric in metric_columns:
        row[f"{metric}_mean"] = part[metric].mean()
        row[f"{metric}_sd"] = part[metric].std(ddof=1)
    aggregate_rows.append(row)

aggregate = pd.DataFrame(aggregate_rows)
aggregate.to_csv(OUT_DIR / "combined_aggregate_v2.csv", index=False)

paired_metrics = [
    "validity",
    "unique_valid_yield",
    "strict_flp_yield",
    "final_candidate_yield",
]
paired_rows = []
for fraction in [50, 100]:
    p0 = combined[
        (combined["family"] == "P0") & (combined["fraction"] == fraction)
    ].set_index("training_seed")
    p5 = combined[
        (combined["family"] == "P5") & (combined["fraction"] == fraction)
    ].set_index("training_seed")

    for training_seed in p0.index:
        row = {"fraction": fraction, "training_seed": training_seed}
        for metric in paired_metrics:
            row[f"{metric}_delta"] = p5.loc[training_seed, metric] - p0.loc[training_seed, metric]
        paired_rows.append(row)

paired = pd.DataFrame(paired_rows)

with ZipFile(ARCHIVE) as archive:
    new_bpc = pd.read_csv(io.BytesIO(archive.read("fine_tuning_summary.csv")))
old_bpc = pd.read_csv(
    LEARNING_DIR / "best_runs.csv"
)
old_bpc = old_bpc[
    old_bpc["family"].isin(["P0", "P5"])
    & old_bpc["fraction"].isin([50, 100])
]
combined_bpc = pd.concat([old_bpc, new_bpc], ignore_index=True)
combined_bpc.to_csv(OUT_DIR / "combined_validation_bpc.csv", index=False)

for fraction in [50, 100]:
    p0 = combined_bpc[
        (combined_bpc["family"] == "P0") & (combined_bpc["fraction"] == fraction)
    ].set_index("training_seed")["best_validation_bpc"]
    p5 = combined_bpc[
        (combined_bpc["family"] == "P5") & (combined_bpc["fraction"] == fraction)
    ].set_index("training_seed")["best_validation_bpc"]
    improvement = p0 - p5
    for training_seed, value in improvement.items():
        paired.loc[
            (paired["fraction"] == fraction)
            & (paired["training_seed"] == training_seed),
            "validation_bpc_improvement",
        ] = value

paired.to_csv(OUT_DIR / "paired_deltas_p5_vs_p0.csv", index=False)

test_rows = []
test_metrics = paired_metrics + ["validation_bpc"]
for fraction in [50, 100]:
    part = paired[paired["fraction"] == fraction]
    for metric in test_metrics:
        column = (
            "validation_bpc_improvement"
            if metric == "validation_bpc"
            else f"{metric}_delta"
        )
        values = part[column]
        wins = int((values > 0).sum())
        test_rows.append({
            "fraction": fraction,
            "metric": metric,
            "seeds": len(values),
            "p5_wins": wins,
            "mean_improvement": values.mean(),
            "median_improvement": values.median(),
            "one_sided_sign_test_p": binomtest(
                wins, len(values), p=0.5, alternative="greater"
            ).pvalue,
        })

tests = pd.DataFrame(test_rows)
tests.to_csv(OUT_DIR / "paired_sign_tests.csv", index=False)

candidate_union = (
    pd.concat(candidate_tables, ignore_index=True)
    .sort_values("snn_to_train", ascending=False)
    .drop_duplicates("canonical_smiles")
)
candidate_union.to_csv(OUT_DIR / "candidate_union_v2.csv", index=False)

colors = {50: "#75658C", 100: "#2F6B4F"}
panels = [
    ("validity_delta", "Δ validity"),
    ("strict_flp_yield_delta", "Δ strict FLP-like"),
    ("final_candidate_yield_delta", "Δ final candidates"),
    ("validation_bpc_improvement", "Улучшение validation BPC"),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
for ax, (column, title) in zip(axes.flat, panels):
    for fraction in [50, 100]:
        part = paired[paired["fraction"] == fraction]
        ax.plot(
            part["training_seed"],
            part[column],
            marker="o",
            linewidth=1.5,
            color=colors[fraction],
            label=str(fraction) + "%",
        )
    ax.axhline(0, color="#555555", linewidth=1)
    ax.axvline(38.5, color="#BBBBBB", linestyle="--", linewidth=1)
    ax.set_xticks([11, 22, 33, 44, 55, 66])
    ax.set_title(title)
    ax.set_xlabel("End-to-end seed")
    ax.grid(alpha=0.2)

axes.flat[0].legend(frameon=False)
fig.suptitle("P5 относительно P0: шесть независимых повторов", fontsize=14)
fig.tight_layout()
fig.savefig(OUT_DIR / "confirmatory_paired_results.png", dpi=220)

print()
print("Evaluator:", EVALUATOR_VERSION)
print(
    aggregate[[
        "family",
        "fraction",
        "validity_mean",
        "strict_flp_yield_mean",
        "final_candidate_yield_mean",
    ]].round(4).to_string(index=False)
)
print()
print(tests.round(4).to_string(index=False))
print("Уникальных новых финальных кандидатов:", len(candidate_union))
