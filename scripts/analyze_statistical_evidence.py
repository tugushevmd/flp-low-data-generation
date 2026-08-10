import argparse
import itertools
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, t


COHORTS = {
    "discovery": [11, 22, 33],
    "confirmatory": [44, 55, 66],
    "pooled": [11, 22, 33, 44, 55, 66],
}
METRICS = [
    "validity",
    "strict_flp_yield",
    "final_candidate_yield",
    "validation_bpc",
]
PRIMARY_METRIC = "strict_flp_yield"
COLORS = {
    "discovery": "#356859",
    "confirmatory": "#963F52",
    "pooled": "#7663A5",
}


def exact_sign_flip(values):
    observed = np.mean(values)
    statistics = [
        np.mean(values * np.array(signs))
        for signs in itertools.product([-1, 1], repeat=len(values))
    ]
    return np.mean(np.array(statistics) >= observed - 1e-12)


def paired_summary(table, metric, fraction, cohort, seeds):
    part = table[
        (table["fraction"] == fraction)
        & (table["training_seed"].isin(seeds))
    ]
    paired = part.pivot(
        index="training_seed",
        columns="family",
        values=metric,
    ).dropna()

    improvement = paired["P0"] - paired["P5"] if "bpc" in metric else paired["P5"] - paired["P0"]
    n = len(improvement)
    standard_error = improvement.std(ddof=1) / np.sqrt(n)
    half_width = t.ppf(0.975, n - 1) * standard_error
    nonzero = improvement[improvement != 0]
    wins = int((nonzero > 0).sum())

    return {
        "cohort": cohort,
        "fraction": fraction,
        "metric": metric,
        "role": "primary" if metric == PRIMARY_METRIC else "secondary",
        "seeds": n,
        "P0_mean": paired["P0"].mean(),
        "P0_sd": paired["P0"].std(ddof=1),
        "P5_mean": paired["P5"].mean(),
        "P5_sd": paired["P5"].std(ddof=1),
        "mean_improvement": improvement.mean(),
        "improvement_ci_low": improvement.mean() - half_width,
        "improvement_ci_high": improvement.mean() + half_width,
        "positive_pairs": wins,
        "sign_test_p": binomtest(
            wins,
            len(nonzero),
            p=0.5,
            alternative="greater",
        ).pvalue,
        "sign_flip_mean_p": exact_sign_flip(improvement.to_numpy()),
    }


def exact_page_test(values):
    matrix = np.asarray(values, dtype=float)
    ranks = np.vstack([rankdata(row) for row in matrix])
    weights = np.arange(1, matrix.shape[1] + 1)
    observed = np.sum(ranks.sum(axis=0) * weights)

    permutations = list(itertools.permutations(range(matrix.shape[1])))
    statistics = []
    for choices in itertools.product(permutations, repeat=matrix.shape[0]):
        permuted = np.vstack([
            ranks[row, order]
            for row, order in enumerate(choices)
        ])
        statistics.append(np.sum(permuted.sum(axis=0) * weights))
    return observed, np.mean(np.asarray(statistics) >= observed)


def page_row(table, metric, fraction, families, increasing=True):
    part = table[
        (table["fraction"] == fraction)
        & (table["training_seed"].isin(COHORTS["discovery"]))
        & (table["family"].isin(families))
    ]
    matrix = part.pivot(
        index="training_seed",
        columns="family",
        values=metric,
    )[families]
    values = matrix.to_numpy()
    if not increasing:
        values = -values
    statistic, p_value = exact_page_test(values)
    monotonic_seeds = sum(
        np.all(np.diff(row) > 0)
        for row in values
    )
    return {
        "analysis": "fine_tuning",
        "metric": metric,
        "fraction": fraction,
        "ordered_levels": " < ".join(families),
        "seeds": len(matrix),
        "strictly_monotonic_seeds": monotonic_seeds,
        "page_statistic": statistic,
        "exact_one_sided_p": p_value,
    }


def read_fixed_step_bpc(learning_archive, confirmatory_archive):
    with ZipFile(learning_archive) as archive:
        discovery = pd.read_csv(archive.open("training_history.csv"))
    with ZipFile(confirmatory_archive) as archive:
        confirmatory = pd.read_csv(archive.open("fine_tuning_history.csv"))
    history = pd.concat([discovery, confirmatory], ignore_index=True)
    history = history[
        (history["step"] == 125)
        & (history["family"].isin(["P0", "P5"]))
        & (history["fraction"].isin([50, 100]))
    ].copy()
    return history.rename(columns={"validation_bpc": "validation_bpc_fixed_8000"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--table-dir", default="results/publication_tables")
    parser.add_argument("--figure-dir", default="figures")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    table_dir = Path(args.table_dir)
    figure_dir = Path(args.figure_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    generated = pd.read_csv(
        result_dir / "controlled_prior_confirmatory/combined_seed_summary_v2.csv"
    )
    bpc = pd.read_csv(
        result_dir / "controlled_prior_confirmatory/combined_validation_bpc.csv"
    )[["family", "fraction", "training_seed", "best_validation_bpc"]].rename(
        columns={"best_validation_bpc": "validation_bpc"}
    )
    combined = generated.merge(
        bpc,
        on=["family", "fraction", "training_seed"],
        how="left",
    )

    effect_rows = []
    for cohort, seeds in COHORTS.items():
        for fraction in [50, 100]:
            for metric in METRICS:
                effect_rows.append(
                    paired_summary(combined, metric, fraction, cohort, seeds)
                )

    fixed = read_fixed_step_bpc(
        result_dir / "raw_controlled/learning_curves/controlled_prior_learning_curves_results.zip",
        result_dir / "raw_controlled/confirmatory/controlled_prior_confirmatory_results.zip",
    )
    for cohort, seeds in COHORTS.items():
        for fraction in [50, 100]:
            effect_rows.append(paired_summary(
                fixed,
                "validation_bpc_fixed_8000",
                fraction,
                cohort,
                seeds,
            ))

    effects = pd.DataFrame(effect_rows)
    effects.to_csv(
        table_dir / "table_s4_discovery_confirmatory_effects.csv",
        index=False,
    )

    fixed_generation = pd.read_csv(
        result_dir / "fixed_checkpoint_sensitivity/training_seed_summary_v2.csv"
    ).assign(fraction=100)
    checkpoint_rows = []
    for rule, table in [
        ("validation BPC", combined),
        ("fixed 8000 exposures", fixed_generation),
    ]:
        for metric in [
            "validity",
            "strict_flp_yield",
            "novel_flp_yield",
            "final_candidate_yield",
        ]:
            row = paired_summary(
                table,
                metric,
                100,
                "pooled",
                COHORTS["pooled"],
            )
            row["checkpoint_rule"] = rule
            checkpoint_rows.append(row)
    checkpoint_effects = pd.DataFrame(checkpoint_rows)
    checkpoint_effects.to_csv(
        table_dir / "table_s17_checkpoint_sensitivity.csv",
        index=False,
    )

    frozen = pd.read_csv(result_dir / "controlled_priors/frozen_bpc_runs.csv")
    frozen_matrix = frozen.pivot(
        index="training_seed",
        columns="prior",
        values="flp_gap",
    )[["P0", "P0.1", "P1", "P5"]]
    frozen_statistic, frozen_p = exact_page_test(-frozen_matrix.to_numpy())
    trend_rows = [{
        "analysis": "frozen_prior",
        "metric": "flp_gap",
        "fraction": np.nan,
        "ordered_levels": "P0 < P0.1 < P1 < P5",
        "seeds": len(frozen_matrix),
        "strictly_monotonic_seeds": sum(
            np.all(np.diff(row) > 0)
            for row in -frozen_matrix.to_numpy()
        ),
        "page_statistic": frozen_statistic,
        "exact_one_sided_p": frozen_p,
    }]

    discovery = pd.read_csv(
        result_dir / "controlled_prior_learning_curves/training_seed_summary_v2.csv"
    )
    discovery_bpc = pd.read_csv(
        result_dir / "controlled_prior_learning_curves/best_runs.csv"
    )[["family", "fraction", "training_seed", "best_validation_bpc"]].rename(
        columns={"best_validation_bpc": "validation_bpc"}
    )
    discovery = discovery.merge(
        discovery_bpc,
        on=["family", "fraction", "training_seed"],
        how="left",
    )
    for fraction in [25, 50, 100]:
        for metric in METRICS:
            trend_rows.append(page_row(
                discovery,
                metric,
                fraction,
                ["P0", "P1", "P5"],
                increasing="bpc" not in metric,
            ))

    trends = pd.DataFrame(trend_rows)
    trends.to_csv(table_dir / "table_s5_dose_response_tests.csv", index=False)

    checkpoints = pd.concat([
        pd.read_csv(result_dir / "controlled_prior_learning_curves/best_runs.csv").assign(
            cohort="discovery"
        ),
        pd.read_csv(result_dir / "controlled_prior_confirmatory/combined_validation_bpc.csv")
        .query("training_seed in [44, 55, 66]")
        .assign(cohort="confirmatory"),
    ], ignore_index=True)
    checkpoint_summary = (
        checkpoints.groupby(["cohort", "family", "fraction"])
        .agg(
            seeds=("training_seed", "nunique"),
            mean_best_step=("best_step", "mean"),
            minimum_best_step=("best_step", "min"),
            maximum_best_step=("best_step", "max"),
            selected_at_final_step=("best_step", lambda values: np.mean(values == 125)),
        )
        .reset_index()
    )
    checkpoint_summary.to_csv(
        table_dir / "table_s6_checkpoint_selection.csv",
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes = axes.flatten()

    forest = effects[
        (effects["metric"] == PRIMARY_METRIC)
        & (effects["fraction"] == 100)
    ]
    y = np.arange(len(forest))
    axes[0].errorbar(
        forest["mean_improvement"] * 100,
        y,
        xerr=np.vstack([
            (forest["mean_improvement"] - forest["improvement_ci_low"]) * 100,
            (forest["improvement_ci_high"] - forest["mean_improvement"]) * 100,
        ]),
        fmt="o",
        capsize=4,
        color="#963F52",
    )
    axes[0].axvline(0, color="#686D70", linestyle="--", linewidth=1)
    axes[0].set_yticks(y, forest["cohort"])
    axes[0].set_xlabel("P5 - P0, percentage points")
    axes[0].set_title("Primary outcome at 166 molecules")

    for seed, row in frozen_matrix.iterrows():
        axes[1].plot(
            [0, 150, 1500, 7500],
            row,
            marker="o",
            alpha=0.45,
            color="#7663A5",
        )
    axes[1].plot(
        [0, 150, 1500, 7500],
        frozen_matrix.mean(),
        marker="o",
        linewidth=3,
        color="#963F52",
        label="mean",
    )
    axes[1].set_xscale("symlog", linthresh=150)
    axes[1].set_xticks([0, 150, 1500, 7500], ["0", "150", "1,500", "7,500"])
    axes[1].set_xlabel("Main-group molecules in pretraining")
    axes[1].set_ylabel("FLP BPC - general BPC")
    axes[1].set_title(f"Frozen-prior dose response (exact p={frozen_p:.5f})")

    selected = effects[
        (effects["cohort"] == "pooled")
        & (effects["fraction"] == 100)
        & (effects["metric"].isin(["validation_bpc", "validation_bpc_fixed_8000"]))
    ].copy()
    selected["label"] = ["selected checkpoint", "fixed 8,000 exposures"]
    axes[2].bar(
        selected["label"],
        selected["mean_improvement"],
        color=["#356859", "#7663A5"],
    )
    axes[2].axhline(0, color="#686D70", linewidth=1)
    axes[2].set_ylabel("P0 BPC - P5 BPC")
    axes[2].set_title("Checkpoint robustness at 166 molecules")
    axes[2].tick_params(axis="x", rotation=15)

    generation_sensitivity = checkpoint_effects[
        checkpoint_effects["metric"] == PRIMARY_METRIC
    ]
    x = np.arange(len(generation_sensitivity))
    axes[3].errorbar(
        x,
        generation_sensitivity["mean_improvement"] * 100,
        yerr=np.vstack([
            (
                generation_sensitivity["mean_improvement"]
                - generation_sensitivity["improvement_ci_low"]
            ) * 100,
            (
                generation_sensitivity["improvement_ci_high"]
                - generation_sensitivity["mean_improvement"]
            ) * 100,
        ]),
        fmt="o",
        capsize=5,
        color="#356859",
    )
    axes[3].axhline(0, color="#686D70", linewidth=1)
    axes[3].set_xticks(x, generation_sensitivity["checkpoint_rule"])
    axes[3].set_ylabel("P5 - P0, percentage points")
    axes[3].set_title("Generation checkpoint sensitivity")
    axes[3].tick_params(axis="x", rotation=15)

    fig.suptitle("Statistical analysis of the controlled-prior experiment", fontsize=15, weight="bold")
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(
            figure_dir / f"figure_s2_statistical_evidence.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)

    print(effects[
        (effects["metric"] == PRIMARY_METRIC)
        & (effects["fraction"] == 100)
    ].round(4).to_string(index=False))
    print()
    print(trends.round(5).to_string(index=False))
    print()
    print(checkpoint_summary.to_string(index=False))
    print()
    print(checkpoint_effects.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
