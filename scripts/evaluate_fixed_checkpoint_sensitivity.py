import argparse
import io
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import audit_generation


def paired_summary(table, rule):
    pivot = table.pivot(
        index="training_seed",
        columns="family",
        values="strict_flp_yield",
    )
    differences = pivot["P5"] - pivot["P0"]
    standard_error = differences.std(ddof=1) / np.sqrt(len(differences))
    interval = t.ppf(0.975, len(differences) - 1) * standard_error
    return {
        "checkpoint_rule": rule,
        "training_seeds": len(differences),
        "mean_P0": pivot["P0"].mean(),
        "mean_P5": pivot["P5"].mean(),
        "mean_paired_difference": differences.mean(),
        "ci95_low": differences.mean() - interval,
        "ci95_high": differences.mean() + interval,
        "P5_wins": int((differences > 0).sum()),
    }


parser = argparse.ArgumentParser()
parser.add_argument(
    "--archive",
    default=(
        "results/raw_controlled/fixed_checkpoint/"
        "controlled_prior_fixed_checkpoint_results.zip"
    ),
)
parser.add_argument(
    "--out-dir",
    default="results/fixed_checkpoint_sensitivity",
)
args = parser.parse_args()

archive_path = Path(args.archive)
if not archive_path.is_absolute():
    archive_path = ROOT / archive_path
out_dir = ROOT / args.out_dir
out_dir.mkdir(parents=True, exist_ok=True)

curation_dir = ROOT / "data" / "flp_curation"
train_smiles = (
    pd.read_csv(ROOT / "data" / "controlled_priors" / "learning_curve_subsets.csv")
    .iloc[:166]["smiles"]
    .astype(str)
    .tolist()
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

pattern = re.compile(
    r"fixed_(P0|P5)_train_(11|22|33|44|55|66)"
    r"_generation_(101|202|303)\.csv"
)
rows = []

with ZipFile(archive_path) as archive:
    sample_names = sorted(
        name for name in archive.namelist() if pattern.fullmatch(name)
    )
    for name in sample_names:
        match = pattern.fullmatch(name)
        samples = pd.read_csv(io.BytesIO(archive.read(name)))
        audit, summary, _ = audit_generation(
            raw_smiles=samples["generated_smiles"].fillna("").tolist(),
            train_smiles=train_smiles,
            seed_smiles=seed_smiles,
            template_smiles=template_smiles,
            reached_max_length=(
                samples["reached_max_length"].fillna(False).astype(bool).tolist()
            ),
        )
        row = summary.iloc[0].to_dict()
        row.update({
            "family": match.group(1),
            "training_seed": int(match.group(2)),
            "generation_seed": int(match.group(3)),
        })
        rows.append(row)

if len(rows) != 36:
    raise ValueError(f"Found {len(rows)} runs; expected 36")

run_table = pd.DataFrame(rows)
run_table.to_csv(out_dir / "run_summary_v2.csv", index=False)

metrics = [
    column
    for column in run_table.columns
    if column not in {
        "evaluator_version",
        "family",
        "training_seed",
        "generation_seed",
    }
]
fixed = (
    run_table.groupby(["family", "training_seed"])[metrics]
    .mean()
    .reset_index()
)
fixed.to_csv(out_dir / "training_seed_summary_v2.csv", index=False)

selected = pd.read_csv(
    ROOT
    / "results"
    / "controlled_prior_confirmatory"
    / "combined_seed_summary_v2.csv"
)
selected = selected[
    selected["family"].isin(["P0", "P5"])
    & selected["fraction"].eq(100)
]

comparison = pd.DataFrame([
    paired_summary(selected, "validation BPC"),
    paired_summary(fixed, "fixed 8000 exposures"),
])
comparison.to_csv(out_dir / "checkpoint_rule_comparison.csv", index=False)

fig, ax = plt.subplots(figsize=(7.2, 4.5))
x = np.arange(len(comparison))
ax.errorbar(
    x,
    comparison["mean_paired_difference"] * 100,
    yerr=[
        (comparison["mean_paired_difference"] - comparison["ci95_low"]) * 100,
        (comparison["ci95_high"] - comparison["mean_paired_difference"]) * 100,
    ],
    fmt="o",
    color="#356859",
    capsize=5,
)
ax.axhline(0, color="#686D70", linewidth=1)
ax.set_xticks(x, comparison["checkpoint_rule"])
ax.set_ylabel("P5 - P0 strict yield, percentage points")
ax.set_title("Checkpoint-selection sensitivity")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(out_dir / "checkpoint_rule_sensitivity.png", dpi=220)
plt.close(fig)

print(comparison.round(4).to_string(index=False))
