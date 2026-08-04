from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluation.evaluator import audit_generation
from flp_design.fragments import enumerate_fragment_library


DATA_DIR = ROOT / "data" / "flp_curation"
CURATION_DIR = DATA_DIR
RESULT_DIR = ROOT / "results" / "baselines" / "fragment_recombination"

GENERATION_SEEDS = [101, 202, 303]
SAMPLES_PER_SEED = 1000


def read_smiles(path, column="smiles"):
    return pd.read_csv(path)[column].dropna().astype(str).tolist()


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_smiles = read_smiles(CURATION_DIR / "train_smiles.csv")
    reference_smiles = read_smiles(
        CURATION_DIR / "reference_smiles_curated_v2.csv",
        "canonical_smiles",
    )

    library = enumerate_fragment_library()
    library.to_csv(RESULT_DIR / "fragment_library.csv", index=False)
    template_smiles = library["canonical_smiles"].tolist()

    old_library = set(read_smiles(
        DATA_DIR / "template_candidates.csv",
        "canonical_smiles",
    ))
    same_library = set(template_smiles) == old_library
    print("Structures in library:", len(library))
    print("Matches the source template library:", same_library)
    print(library.groupby("lb_atom").size().to_string())

    full_audit, full_summary, full_funnel = audit_generation(
        raw_smiles=template_smiles,
        train_smiles=train_smiles,
        seed_smiles=reference_smiles,
        template_smiles=template_smiles,
    )
    full_audit["is_fragment_candidate"] = (
        full_audit["is_strict_flp_like"]
        & full_audit["novel_vs_all_seed"]
        & full_audit["snn_to_train"].between(0.40, 0.90)
        & (full_audit["snn_to_train"] < 0.95)
    )
    full_audit = pd.concat(
        [library.drop(columns="canonical_smiles"), full_audit],
        axis=1,
    )
    full_summary["fragment_candidate_n"] = int(
        full_audit["is_fragment_candidate"].sum()
    )
    full_summary["fragment_candidate_yield"] = (
        full_audit["is_fragment_candidate"].mean()
    )
    full_audit.to_csv(
        RESULT_DIR / "full_library_evaluated.csv",
        index=False,
    )
    full_summary.to_csv(
        RESULT_DIR / "full_library_summary.csv",
        index=False,
    )
    full_funnel.to_csv(
        RESULT_DIR / "full_library_funnel.csv",
        index=False,
    )
    family_summary = (
        full_audit
        .groupby(["lb_atom", "linker"])
        .agg(
            n=("canonical_smiles", "size"),
            candidate_yield=("is_fragment_candidate", "mean"),
            mean_snn=("snn_to_train", "mean"),
        )
        .reset_index()
    )
    family_summary.to_csv(
        RESULT_DIR / "family_summary.csv",
        index=False,
    )

    summaries = []
    candidate_tables = []

    for seed in GENERATION_SEEDS:
        sample = library.sample(
            n=SAMPLES_PER_SEED,
            replace=False,
            random_state=seed,
        ).reset_index(drop=True)

        audit, summary, funnel = audit_generation(
            raw_smiles=sample["canonical_smiles"],
            train_smiles=train_smiles,
            seed_smiles=reference_smiles,
            template_smiles=template_smiles,
        )
        audit = pd.concat(
            [sample.drop(columns="canonical_smiles"), audit],
            axis=1,
        )

        audit["is_fragment_candidate"] = (
            audit["is_strict_flp_like"]
            & audit["novel_vs_all_seed"]
            & audit["snn_to_train"].between(0.40, 0.90)
            & (audit["snn_to_train"] < 0.95)
        )

        row = summary.iloc[0].to_dict()
        row["generation_seed"] = seed
        row["fragment_candidate_n"] = int(
            audit["is_fragment_candidate"].sum()
        )
        row["fragment_candidate_yield"] = (
            audit["is_fragment_candidate"].mean()
        )
        summaries.append(row)

        candidates = audit[audit["is_fragment_candidate"]].copy()
        candidates["generation_seed"] = seed
        candidate_tables.append(candidates)

        audit.to_csv(
            RESULT_DIR / f"evaluated_seed_{seed}.csv",
            index=False,
        )
        funnel.to_csv(
            RESULT_DIR / f"funnel_seed_{seed}.csv",
            index=False,
        )

        print(
            "seed", seed,
            "| strict FLP", round(row["strict_flp_yield"], 3),
            "| fragment candidates",
            round(row["fragment_candidate_yield"], 3),
        )

    run_summary = pd.DataFrame(summaries)
    run_summary.to_csv(RESULT_DIR / "run_summary.csv", index=False)

    metrics = [
        "validity",
        "unique_valid_yield",
        "chemically_sane_fraction",
        "strict_flp_yield",
        "novel_flp_yield",
        "final_candidate_yield",
        "fragment_candidate_yield",
        "near_duplicate_095",
        "mean_snn_to_train",
        "internal_diversity",
        "scaffold_novelty_vs_train",
    ]
    aggregate = pd.DataFrame({
        "metric": metrics,
        "mean": [run_summary[metric].mean() for metric in metrics],
        "sd": [run_summary[metric].std() for metric in metrics],
    })
    aggregate.to_csv(RESULT_DIR / "aggregate_summary.csv", index=False)

    all_candidates = pd.concat(candidate_tables, ignore_index=True)
    candidate_counts = (
        all_candidates["canonical_smiles"]
        .value_counts()
        .rename("sample_count")
        .reset_index()
    )
    candidate_union = (
        all_candidates
        .drop_duplicates("canonical_smiles")
        .merge(candidate_counts, on="canonical_smiles")
    )
    candidate_union.to_csv(
        RESULT_DIR / "fragment_candidate_union.csv",
        index=False,
    )

    plot_metrics = [
        ("validity", "Valid"),
        ("chemically_sane_fraction", "Chemically sane"),
        ("strict_flp_yield", "Strict FLP-like"),
        ("fragment_candidate_yield", "Candidates outside curated set"),
    ]
    means = [run_summary[metric].mean() for metric, _ in plot_metrics]
    errors = [run_summary[metric].std() for metric, _ in plot_metrics]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(
        [label for _, label in plot_metrics],
        means,
        yerr=errors,
        color=["#356859", "#7B6D8D", "#8C3B4A", "#66734E"],
        capsize=4,
    )
    ax.set_ylabel("Fraction of all generations")
    ax.set_title("Fragment recombination baseline")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        RESULT_DIR / "fragment_recombination_summary.png",
        dpi=180,
    )

    print()
    print(aggregate.round(4).to_string(index=False))
    print("Unique fragment candidates:", len(candidate_union))
    print(
        "Candidates in the full library:",
        int(full_summary.at[0, "fragment_candidate_n"]),
    )


if __name__ == "__main__":
    main()
