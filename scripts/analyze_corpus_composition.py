import argparse
import gzip
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from rdkit import Chem


PRIORS = ["P0", "P0.1", "P1", "P5"]
FEATURES = [
    "smiles_length",
    "token_count",
    "branch_count",
    "heavy_atoms",
    "ring_count",
    "aromatic_rings",
    "rotatable_bonds",
]
TOKEN_PATTERN = re.compile(
    r"\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|"
    r"\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%\d{2}|\d"
)
COLORS = {
    "P0": "#686D70",
    "P0.1": "#356859",
    "P1": "#7663A5",
    "P5": "#963F52",
}


def read_corpus(archive, prior):
    with gzip.GzipFile(fileobj=archive.open(f"{prior}.csv.gz")) as handle:
        table = pd.read_csv(handle)
    table["token_count"] = table["smiles"].map(
        lambda value: len(TOKEN_PATTERN.findall(value))
    )
    table["branch_count"] = table["smiles"].str.count(r"\(")
    return table


def scaffold_statistics(values):
    frequencies = values.fillna("<acyclic>").value_counts(normalize=True)
    entropy = -(frequencies * np.log(frequencies)).sum()
    return {
        "unique_scaffolds": len(frequencies),
        "unique_scaffold_fraction": len(frequencies) / len(values),
        "effective_scaffolds": np.exp(entropy),
        "top_10_scaffold_fraction": frequencies.head(10).sum(),
    }


def token_frequencies(smiles):
    counts = Counter()
    for value in smiles:
        counts.update(TOKEN_PATTERN.findall(value))
    return counts


def token_divergence(reference, observed):
    tokens = sorted(set(reference) | set(observed))
    ref = np.array([reference[token] + 0.5 for token in tokens], dtype=float)
    obs = np.array([observed[token] + 0.5 for token in tokens], dtype=float)
    ref /= ref.sum()
    obs /= obs.sum()
    midpoint = (ref + obs) / 2
    return {
        "kl_from_P0": float(np.sum(obs * np.log(obs / ref))),
        "js_divergence": float(
            0.5 * np.sum(ref * np.log(ref / midpoint))
            + 0.5 * np.sum(obs * np.log(obs / midpoint))
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        default="data/controlled_priors/controlled_prior_corpora.zip",
    )
    parser.add_argument(
        "--curated-reference",
        default="data/flp_curation/reference_smiles_curated_v2.csv",
    )
    parser.add_argument("--table-dir", default="results/publication_tables")
    parser.add_argument("--figure-dir", default="figures")
    args = parser.parse_args()

    table_dir = Path(args.table_dir)
    figure_dir = Path(args.figure_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    corpora = {}
    tokens = {}
    summary_rows = []

    with ZipFile(args.archive) as archive:
        for prior in PRIORS:
            table = read_corpus(archive, prior)
            corpora[prior] = table
            tokens[prior] = token_frequencies(table["smiles"])

            row = {"prior": prior, "molecules": len(table)}
            for feature in FEATURES:
                row[f"{feature}_mean"] = table[feature].mean()
                row[f"{feature}_sd"] = table[feature].std(ddof=1)
            row.update(scaffold_statistics(table["scaffold"]))
            summary_rows.append(row)

        with gzip.GzipFile(
            fileobj=archive.open("selected_main_group.csv.gz")
        ) as handle:
            selected_main_group = pd.read_csv(handle)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(table_dir / "table_s1_corpus_composition.csv", index=False)

    p0 = corpora["P0"]
    comparison_rows = []
    for prior in PRIORS[1:]:
        current = corpora[prior]
        divergence = token_divergence(tokens["P0"], tokens[prior])
        for feature in FEATURES:
            pooled_sd = np.sqrt(
                (p0[feature].var(ddof=1) + current[feature].var(ddof=1)) / 2
            )
            comparison_rows.append({
                "prior": prior,
                "feature": feature,
                "P0_mean": p0[feature].mean(),
                "prior_mean": current[feature].mean(),
                "mean_difference": current[feature].mean() - p0[feature].mean(),
                "standardized_mean_difference": (
                    (current[feature].mean() - p0[feature].mean()) / pooled_sd
                ),
                **divergence,
            })

    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(
        table_dir / "table_s2_composition_differences.csv",
        index=False,
    )

    all_tokens = sorted(set(tokens["P0"]) | set(tokens["P5"]))
    p0_total = sum(tokens["P0"].values())
    p5_total = sum(tokens["P5"].values())
    token_shifts = pd.DataFrame({
        "token": all_tokens,
        "P0_frequency": [tokens["P0"][token] / p0_total for token in all_tokens],
        "P5_frequency": [tokens["P5"][token] / p5_total for token in all_tokens],
    })
    token_shifts["frequency_difference"] = (
        token_shifts["P5_frequency"] - token_shifts["P0_frequency"]
    )
    token_shifts["absolute_difference"] = token_shifts[
        "frequency_difference"
    ].abs()
    token_shifts.sort_values("absolute_difference", ascending=False).to_csv(
        table_dir / "table_s3_token_shifts.csv",
        index=False,
    )

    element_sets = selected_main_group["elements"].str.split(",").map(set)
    query_counts = selected_main_group["query_element"].value_counts()
    element_rows = []
    for element in ["B", "P", "N", "Si", "Al", "Ge"]:
        presence_n = int(element_sets.map(lambda values: element in values).sum())
        query_n = int(query_counts.get(element, 0))
        element_rows.append({
            "feature": element,
            "query_selection_n": query_n,
            "query_selection_fraction": query_n / len(selected_main_group),
            "molecule_presence_n": presence_n,
            "molecule_presence_fraction": presence_n / len(selected_main_group),
        })

    b_with_base = element_sets.map(
        lambda values: "B" in values and bool({"P", "N"} & values)
    ).sum()
    element_rows.append({
        "feature": "B + (P or N)",
        "query_selection_n": np.nan,
        "query_selection_fraction": np.nan,
        "molecule_presence_n": int(b_with_base),
        "molecule_presence_fraction": b_with_base / len(selected_main_group),
    })
    element_table = pd.DataFrame(element_rows)
    element_table.to_csv(
        table_dir / "table_s13_main_group_element_composition.csv",
        index=False,
    )

    curated = pd.read_csv(args.curated_reference)
    curated_elements = curated["canonical_smiles"].map(
        lambda value: {
            atom.GetSymbol()
            for atom in Chem.MolFromSmiles(value).GetAtoms()
        }
    )
    curated_rows = []
    for label, condition in [
        ("B-containing", lambda values: "B" in values),
        ("non-B, Al-containing", lambda values: "B" not in values and "Al" in values),
        ("non-B, Al- and Si-containing", lambda values: "B" not in values and {"Al", "Si"} <= values),
    ]:
        count = int(curated_elements.map(condition).sum())
        curated_rows.append({
            "group": label,
            "molecules": count,
            "fraction": count / len(curated),
        })
    pd.DataFrame(curated_rows).to_csv(
        table_dir / "table_s16_curated_reference_scope.csv",
        index=False,
    )

    plt.rcParams.update({
        "figure.dpi": 120,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "legend.frameon": False,
    })
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.3))

    smd = comparisons.pivot(
        index="feature",
        columns="prior",
        values="standardized_mean_difference",
    ).loc[FEATURES]
    smd_limit = smd.abs().to_numpy().max()
    image = axes[0].imshow(
        smd.abs(),
        aspect="auto",
        cmap="Greys",
        vmin=0,
        vmax=smd_limit,
    )
    axes[0].set_xticks(range(3), PRIORS[1:])
    axes[0].set_yticks(range(len(FEATURES)), [name.replace("_", " ") for name in FEATURES])
    axes[0].set_title("Absolute SMD versus P0")
    axes[0].grid(False)
    for row in range(len(FEATURES)):
        for column in range(3):
            axes[0].text(
                column,
                row,
                f"{smd.abs().iloc[row, column]:.3f}",
                ha="center",
                va="center",
                color="white" if smd.abs().iloc[row, column] > smd_limit / 2 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    divergence = comparisons.drop_duplicates("prior")
    axes[1].bar(
        divergence["prior"],
        divergence["js_divergence"],
        color=[COLORS[prior] for prior in divergence["prior"]],
    )
    axes[1].set_title("SMILES-token divergence")
    axes[1].set_ylabel("Jensen-Shannon divergence")

    scaffold_columns = [
        "unique_scaffold_fraction",
        "top_10_scaffold_fraction",
    ]
    x = np.arange(len(PRIORS))
    width = 0.34
    for offset, column, label, color in [
        (-width / 2, scaffold_columns[0], "Unique scaffold fraction", COLORS["P0.1"]),
        (width / 2, scaffold_columns[1], "Top-10 scaffold fraction", COLORS["P5"]),
    ]:
        axes[2].bar(
            x + offset,
            summary[column],
            width,
            label=label,
            color=color,
            alpha=0.85,
        )
    axes[2].set_xticks(x, PRIORS)
    axes[2].set_title("Scaffold distribution")
    axes[2].set_ylabel("Fraction of corpus")
    axes[2].legend(fontsize=8)

    plotted_elements = element_table.iloc[:6]
    y = np.arange(len(plotted_elements))
    height = 0.34
    axes[3].barh(
        y - height / 2,
        plotted_elements["query_selection_fraction"],
        height,
        color=COLORS["P1"],
        label="Selection target",
    )
    axes[3].barh(
        y + height / 2,
        plotted_elements["molecule_presence_fraction"],
        height,
        color=COLORS["P5"],
        label="Present in molecule",
    )
    axes[3].set_yticks(y, plotted_elements["feature"])
    axes[3].invert_yaxis()
    axes[3].xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[3].set_title("Selected main-group component")
    axes[3].set_xlabel("Fraction of 7,500 molecules")
    axes[3].legend(fontsize=8)

    fig.suptitle("Controlled-prior corpus composition audit", fontsize=15, weight="bold")
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(
            figure_dir / f"figure_s1_corpus_composition_audit.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)

    print(summary.round(4).to_string(index=False))
    print()
    print(comparisons.groupby("prior")["standardized_mean_difference"].apply(
        lambda values: values.abs().max()
    ).rename("maximum absolute SMD").round(4).to_string())
    print()
    print(divergence[["prior", "kl_from_P0", "js_divergence"]].round(6).to_string(index=False))
    print()
    print(element_table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
