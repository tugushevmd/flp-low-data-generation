import io
import re
import sys
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import audit_generation
from evaluation.evaluator import murcko_smiles, nearest_similarities


RESULTS = ROOT / "results"
TABLES = RESULTS / "publication_tables"
FIGURES = ROOT / "figures"

ARCHIVES = [
    (
        RESULTS / "raw_controlled" / "learning_curves"
        / "controlled_prior_learning_curves_results.zip",
        re.compile(
            r"learning_curve_(P0|P5)_fraction_100"
            r"_train_(11|22|33)_generation_(101|202|303)\.csv"
        ),
    ),
    (
        RESULTS / "raw_controlled" / "confirmatory"
        / "controlled_prior_confirmatory_results.zip",
        re.compile(
            r"confirmatory_(P0|P5)_fraction_100"
            r"_train_(44|55|66)_generation_(101|202|303)\.csv"
        ),
    ),
]

curation = ROOT / "data" / "flp_curation"
train_smiles = pd.read_csv(
    ROOT / "data" / "controlled_priors" / "learning_curve_subsets.csv"
)["smiles"].iloc[:166].astype(str).tolist()
seed_smiles = pd.read_csv(
    curation / "reference_smiles_curated_v2.csv"
)["canonical_smiles"].dropna().astype(str).tolist()
template_smiles = pd.read_csv(
    RESULTS / "baselines" / "fragment_recombination" / "fragment_library.csv"
)["canonical_smiles"].dropna().astype(str).tolist()

candidate_tables = []
for archive_path, pattern in ARCHIVES:
    with ZipFile(archive_path) as archive:
        for name in sorted(archive.namelist()):
            match = pattern.fullmatch(name)
            if match is None:
                continue

            samples = pd.read_csv(io.BytesIO(archive.read(name)))
            audit, _, _ = audit_generation(
                raw_smiles=samples["generated_smiles"].fillna("").tolist(),
                train_smiles=train_smiles,
                seed_smiles=seed_smiles,
                template_smiles=template_smiles,
                reached_max_length=(
                    samples["reached_max_length"].fillna(False).astype(bool).tolist()
                ),
            )
            candidates = audit.loc[
                audit["is_final_candidate"], ["canonical_smiles"]
            ].copy()
            candidates["family"] = match.group(1)
            candidates["training_seed"] = int(match.group(2))
            candidates["generation_seed"] = int(match.group(3))
            candidate_tables.append(candidates)

candidates = pd.concat(candidate_tables, ignore_index=True)
occurrence_counts = candidates.groupby("family").size()
candidates = candidates.drop_duplicates(["family", "canonical_smiles"])

template_set = set(template_smiles)
template_scaffolds = {
    scaffold for scaffold in map(murcko_smiles, template_smiles) if scaffold
}

proximity_tables = []
for family, part in candidates.groupby("family"):
    part = part.copy()
    part["nearest_template_tanimoto"] = nearest_similarities(
        part["canonical_smiles"].tolist(), template_smiles
    )
    part["exact_template_match"] = part["canonical_smiles"].isin(template_set)
    part["scaffold_in_template_library"] = part["canonical_smiles"].map(
        lambda smiles: murcko_smiles(smiles) in template_scaffolds
    )
    proximity_tables.append(part)

proximity = pd.concat(proximity_tables, ignore_index=True)

summary_rows = []
for family, part in proximity.groupby("family"):
    similarities = part["nearest_template_tanimoto"]
    summary_rows.append({
        "family": family,
        "candidate_occurrences": int(occurrence_counts[family]),
        "unique_candidates": len(part),
        "mean_nearest_template_tanimoto": similarities.mean(),
        "median_nearest_template_tanimoto": similarities.median(),
        "q25": similarities.quantile(0.25),
        "q75": similarities.quantile(0.75),
        "fraction_at_least_0.80": (similarities >= 0.80).mean(),
        "fraction_at_least_0.90": (similarities >= 0.90).mean(),
        "fraction_at_least_0.95": (similarities >= 0.95).mean(),
        "exact_template_match_fraction": part["exact_template_match"].mean(),
        "template_scaffold_overlap": part["scaffold_in_template_library"].mean(),
    })

summary = pd.DataFrame(summary_rows).sort_values("family")
summary.to_csv(
    TABLES / "table_s15_fragment_library_proximity.csv",
    index=False,
)

colors = {"P0": "#555A61", "P5": "#963F52"}
fig, ax = plt.subplots(figsize=(7.2, 4.5))
for family in ["P0", "P5"]:
    values = np.sort(
        proximity.loc[
            proximity["family"] == family,
            "nearest_template_tanimoto",
        ].to_numpy()
    )
    fraction_at_or_above = 1 - np.arange(len(values)) / len(values)
    ax.step(
        values,
        fraction_at_or_above,
        where="post",
        linewidth=2.2,
        color=colors[family],
        label=family,
    )

for threshold in [0.8, 0.9, 0.95]:
    ax.axvline(threshold, color="#A8ADB3", linewidth=0.9, linestyle="--")

ax.set_xlim(0.2, 1.0)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Nearest Tanimoto similarity to fragment library")
ax.set_ylabel("Fraction of unique final candidates at or above")
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.set_title("Proximity of controlled-GRU candidates to the rule-based library")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#E7E8EA", linewidth=0.8)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "figure_s5_fragment_library_proximity.png", dpi=320)
fig.savefig(FIGURES / "figure_s5_fragment_library_proximity.pdf")
plt.close(fig)

print(summary.round(4).to_string(index=False))
