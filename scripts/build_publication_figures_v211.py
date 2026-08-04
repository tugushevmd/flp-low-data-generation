from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from rdkit import Chem
from rdkit.Chem import Draw


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results"
TABLE_DIR = ROOT / "results" / "publication_tables"
FIGURE_DIR = ROOT / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "green": "#356B52",
    "wine": "#8A3F50",
    "violet": "#75658C",
    "graphite": "#555A61",
    "gray": "#A8ADB3",
    "light": "#E7E8EA",
}

MODEL_COLORS = {
    "GP-MoLFormer": COLORS["green"],
    "MolGPT": COLORS["wine"],
    "REINVENT": COLORS["violet"],
}

PRIOR_COLORS = {
    "scratch": COLORS["gray"],
    "P0": COLORS["graphite"],
    "P0.1": COLORS["green"],
    "P1": COLORS["violet"],
    "P5": COLORS["wine"],
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.titleweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#4B4F54",
    "axes.linewidth": 0.8,
    "xtick.color": "#33363A",
    "ytick.color": "#33363A",
    "text.color": "#202225",
    "legend.frameon": False,
    "savefig.facecolor": "white",
})


def save_figure(fig, name):
    fig.savefig(FIGURE_DIR / f"{name}.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def finish_axis(ax, percent=False, decimals=0):
    ax.grid(axis="y", color=COLORS["light"], linewidth=0.8)
    ax.set_axisbelow(True)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=decimals))


def panel_letter(ax, letter):
    ax.text(-0.13, 1.08, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")


def mean_and_sd(table, groups, metric):
    return table.groupby(groups)[metric].agg(["mean", "std"]).reset_index()


# 1. Prior compatibility with the FLP domain
prior_seeds = pd.read_csv(DATA / "controlled_priors" / "training_seed_summary_v2.csv")
prior_bpc = pd.read_csv(DATA / "controlled_priors" / "frozen_bpc_runs.csv")
prior_order = ["P0", "P0.1", "P1", "P5"]
x_prior = np.arange(len(prior_order))

fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7))

for seed, part in prior_bpc.groupby("training_seed"):
    part = part.set_index("prior").loc[prior_order]
    axes[0].plot(x_prior, part["flp_gap"], color=COLORS["gray"],
                 linewidth=1, alpha=0.65)
means = prior_bpc.groupby("prior")["flp_gap"].mean().reindex(prior_order)
axes[0].plot(x_prior, means, color=COLORS["graphite"], linewidth=2.2, zorder=3)
for x, prior in zip(x_prior, prior_order):
    values = prior_bpc.loc[prior_bpc["prior"] == prior, "flp_gap"]
    axes[0].scatter(np.full(len(values), x), values, s=34,
                    color=PRIOR_COLORS[prior], edgecolor="white", linewidth=0.6, zorder=4)
axes[0].set_title("FLP BPC gap")
axes[0].set_ylabel("FLP BPC - general BPC")

metrics = [
    ("main_group_fraction_unique_valid", "Main-group atoms", 1),
    ("strict_flp_yield", "Strict FLP-like", 2),
]
for metric, title, axis_number in metrics:
    ax = axes[axis_number]
    for seed, part in prior_seeds.groupby("training_seed"):
        part = part.set_index("prior").loc[prior_order]
        ax.plot(x_prior, part[metric], color=COLORS["gray"], linewidth=1, alpha=0.65)
    means = prior_seeds.groupby("prior")[metric].mean().reindex(prior_order)
    ax.plot(x_prior, means, color=COLORS["graphite"], linewidth=2.2, zorder=3)
    for x, prior in zip(x_prior, prior_order):
        values = prior_seeds.loc[prior_seeds["prior"] == prior, metric]
        ax.scatter(np.full(len(values), x), values, s=34,
                   color=PRIOR_COLORS[prior], edgecolor="white", linewidth=0.6, zorder=4)
    ax.set_title(title)
    ax.set_ylabel("Fraction of valid molecules" if axis_number == 1 else "Fraction of generations")
    finish_axis(ax, percent=True)

for letter, ax in zip("ABC", axes):
    ax.set_xticks(x_prior, prior_order)
    ax.set_xlabel("Prior")
    finish_axis(ax, percent=False if ax is axes[0] else True)
    panel_letter(ax, letter)
axes[2].yaxis.set_major_formatter(PercentFormatter(1, decimals=2))

fig.suptitle("Main-group content improves prior compatibility with the FLP domain",
             fontsize=14, fontweight="semibold", y=1.04)
fig.tight_layout()
save_figure(fig, "figure_1_prior_compatibility")


# 2. Controlled-prior learning curves
curve_seeds = pd.read_csv(DATA / "controlled_prior_learning_curves" / "training_seed_summary_v2.csv")
curve_bpc = pd.read_csv(DATA / "controlled_prior_learning_curves" / "best_runs.csv")
fractions = [25, 50, 100]
molecules = {25: 42, 50: 83, 100: 166}
families = ["scratch", "P0", "P1", "P5"]

panels = [
    ("validity", "Validity", True),
    ("strict_flp_yield", "Strict FLP-like yield", True),
    ("final_candidate_yield", "Final-candidate yield", True),
    ("best_validation_bpc", "Validation BPC", False),
]

fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
for ax, (metric, title, percent), letter in zip(axes.flat, panels, "ABCD"):
    source = curve_bpc if metric == "best_validation_bpc" else curve_seeds
    for family in families:
        family_data = source[source["family"] == family]
        for seed, part in family_data.groupby("training_seed"):
            part = part.sort_values("fraction")
            ax.plot(part["fraction"].map(molecules), part[metric],
                    color=PRIOR_COLORS[family], linewidth=0.9, alpha=0.25)
        average = family_data.groupby("fraction")[metric].mean().reindex(fractions)
        ax.plot([molecules[x] for x in fractions], average,
                color=PRIOR_COLORS[family], marker="o", markersize=5,
                linewidth=2.2, label=family)
    ax.set_title(title)
    ax.set_xticks([42, 83, 166], ["42\n(25%)", "83\n(50%)", "166\n(100%)"])
    ax.set_xlabel("Number of FLP training molecules")
    finish_axis(ax, percent=percent)
    panel_letter(ax, letter)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01))
fig.suptitle("Learning curves by prior and FLP training-set size",
             fontsize=14, fontweight="semibold", y=1.07)
fig.tight_layout()
save_figure(fig, "figure_2_controlled_prior_learning_curves")


# 3. Confirmatory paired experiment: P5 versus P0
deltas = pd.read_csv(DATA / "controlled_prior_confirmatory" / "paired_deltas_p5_vs_p0.csv")
sign_tests = pd.read_csv(DATA / "controlled_prior_confirmatory" / "paired_sign_tests.csv")

delta_panels = [
    ("validity_delta", "Validity", "validity"),
    ("strict_flp_yield_delta", "Strict FLP-like yield", "strict_flp_yield"),
    ("final_candidate_yield_delta", "Final-candidate yield", "final_candidate_yield"),
    ("validation_bpc_improvement", "Validation BPC improvement", "validation_bpc"),
]

fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2))
jitter = np.linspace(-0.12, 0.12, 6)
for ax, (column, title, test_name), letter in zip(axes.flat, delta_panels, "ABCD"):
    for x, fraction in enumerate([50, 100]):
        values = deltas.loc[deltas["fraction"] == fraction, column].to_numpy()
        ax.scatter(x + jitter[:len(values)], values, color=COLORS["violet"],
                   s=38, alpha=0.82, edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter(x, values.mean(), marker="D", s=65, color=COLORS["wine"],
                   edgecolor="white", linewidth=0.7, zorder=4)
        row = sign_tests[(sign_tests["fraction"] == fraction) &
                         (sign_tests["metric"] == test_name)].iloc[0]
        ax.text(x, values.max() + 0.008,
                f"{int(row.p5_wins)}/6", ha="center", va="bottom", fontsize=9,
                color=COLORS["graphite"])
    ax.axhline(0, color=COLORS["graphite"], linewidth=1, linestyle="--")
    ax.set_title(title)
    ax.set_xticks([0, 1], ["83 molecules\n(50%)", "166 molecules\n(100%)"])
    ax.set_ylabel("P5 - P0")
    finish_axis(ax, percent=column != "validation_bpc_improvement")
    panel_letter(ax, letter)

fig.legend([
    plt.Line2D([], [], marker="o", linestyle="", color=COLORS["violet"], markersize=6),
    plt.Line2D([], [], marker="D", linestyle="", color=COLORS["wine"], markersize=6),
], ["individual training seed", "mean"], loc="upper center", ncol=2,
   bbox_to_anchor=(0.5, 1.01))
fig.suptitle("Confirmatory experiment: paired effect of P5 versus P0",
             fontsize=14, fontweight="semibold", y=1.07)
fig.tight_layout()
save_figure(fig, "figure_3_confirmatory_paired_effects")


# 4. Independent external generators
external_parts = []
for model, folder in [
    ("GP-MoLFormer", "external_gpmolformer"),
    ("MolGPT", "external_molgpt"),
    ("REINVENT", "external_reinvent"),
]:
    part = pd.read_csv(DATA / folder / "training_seed_summary_v2.csv")
    part["model"] = model
    external_parts.append(part)
external_seeds = pd.concat(external_parts, ignore_index=True)
zero_shot = pd.read_csv(DATA / "external_zero_shot" / "zero_shot_aggregate_v2.csv")
zero_shot = zero_shot[zero_shot["fraction"] == 100]

external_metrics = [
    ("validity", "Validity"),
    ("strict_flp_yield", "Strict FLP-like yield"),
    ("final_candidate_yield", "Final-candidate yield"),
]

fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9))
for ax, (metric, title), letter in zip(axes, external_metrics, "ABC"):
    for model in MODEL_COLORS:
        color = MODEL_COLORS[model]
        zero = zero_shot[(zero_shot["model"] == model) &
                         (zero_shot["metric"] == metric)].iloc[0]
        model_data = external_seeds[external_seeds["model"] == model]
        means = model_data.groupby("fraction")[metric].mean().reindex([25, 100])
        x = np.array([0, 42, 166])
        y = np.array([zero["mean"], means.loc[25], means.loc[100]])
        ax.plot(x, y, color=color, linewidth=2.1, marker="o", markersize=5, label=model)
        ax.scatter([0], [zero["mean"]], s=70, facecolor="white",
                   edgecolor=color, linewidth=1.8, zorder=4)
        for fraction in [25, 100]:
            values = model_data.loc[model_data["fraction"] == fraction, metric]
            x_value = molecules[fraction]
            ax.scatter(np.full(len(values), x_value), values, color=color,
                       alpha=0.42, s=26, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_title(title)
    ax.set_xticks([0, 42, 166], ["zero-shot", "42\n(25%)", "166\n(100%)"])
    ax.set_xlabel("FLP molecules used for fine-tuning")
    finish_axis(ax, percent=True)
    panel_letter(ax, letter)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
fig.suptitle("Domain adaptation of three pretrained generators",
             fontsize=14, fontweight="semibold", y=1.08)
fig.tight_layout()
save_figure(fig, "figure_4_external_generators")


# 5. Independent manual review of candidates
review_dir = ROOT / "results" / "manual_validation"
review_models = pd.read_csv(review_dir / "review_summary_by_model.csv")
review_overall = pd.read_csv(review_dir / "review_summary_overall.csv").iloc[0]
review_models["order"] = review_models["model"].map({
    "GP-MoLFormer": 2, "MolGPT": 1, "REINVENT": 0,
})
review_models = review_models.sort_values("order")

fig, ax = plt.subplots(figsize=(7.2, 3.6))
for y, row in enumerate(review_models.itertuples()):
    color = MODEL_COLORS[row.model]
    ax.errorbar(row.acceptance_fraction, y,
                xerr=[[row.acceptance_fraction - row.ci95_low],
                      [row.ci95_high - row.acceptance_fraction]],
                fmt="o", markersize=8, color=color, ecolor=color,
                elinewidth=2, capsize=4)
    ax.text(min(row.ci95_high + 0.018, 1.015), y,
            f"{row.accept}/{row.reviewed}", va="center", fontsize=9)
ax.axvline(review_overall.acceptance_fraction, color=COLORS["graphite"],
           linewidth=1, linestyle="--", alpha=0.7)
ax.set_yticks(range(len(review_models)), review_models["model"])
ax.set_xlim(0.68, 1.03)
ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.set_xlabel("Fraction accepted in blinded review")
ax.set_title("Manual chemical review: 68 of 72 structures accepted", fontsize=12)
finish_axis(ax)
fig.tight_layout()
save_figure(fig, "figure_5_manual_chemical_validation")


# 6. Representative structures from the blinded review
review = pd.read_csv(review_dir / "review_results.csv")
accepted = review[review["decision"] == "accept"]
examples = (accepted.sort_values("review_id")
            .groupby(["model", "fraction"], sort=False)
            .head(2)
            .sort_values(["model", "fraction", "review_id"]))

mols = [Chem.MolFromSmiles(smiles) for smiles in examples["canonical_smiles"]]
legends = [
    f"{row.model} | {molecules[int(row.fraction)]} mol | {row.review_id}"
    for row in examples.itertuples()
]
grid = Draw.MolsToGridImage(
    mols,
    molsPerRow=4,
    subImgSize=(360, 285),
    legends=legends,
    useSVG=False,
)
grid.save(FIGURE_DIR / "figure_6_representative_candidates.png")
svg = Draw.MolsToGridImage(
    mols,
    molsPerRow=4,
    subImgSize=(360, 285),
    legends=legends,
    useSVG=True,
)
(FIGURE_DIR / "figure_6_representative_candidates.svg").write_text(svg, encoding="utf-8")


# Tables used in the figures and manuscript
prior_table = prior_seeds.groupby("prior").agg(
    validity_mean=("validity", "mean"),
    main_group_fraction_mean=("main_group_fraction_unique_valid", "mean"),
    strict_flp_yield_mean=("strict_flp_yield", "mean"),
).reindex(prior_order).reset_index()
bpc_table = prior_bpc.groupby("prior").agg(
    flp_bpc_gap_mean=("flp_gap", "mean"),
    flp_bpc_gap_sd=("flp_gap", "std"),
).reindex(prior_order).reset_index()
prior_table.merge(bpc_table, on="prior").to_csv(
    TABLE_DIR / "table_1_prior_compatibility_v211.csv", index=False)

curve_table = mean_and_sd(
    curve_seeds, ["family", "fraction"], "final_candidate_yield")
curve_table = curve_table.rename(columns={"mean": "final_yield_mean", "std": "final_yield_sd"})
for metric in ["validity", "strict_flp_yield"]:
    part = mean_and_sd(curve_seeds, ["family", "fraction"], metric)
    part = part.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_sd"})
    curve_table = curve_table.merge(part, on=["family", "fraction"])
bpc_means = mean_and_sd(curve_bpc, ["family", "fraction"], "best_validation_bpc")
bpc_means = bpc_means.rename(columns={"mean": "validation_bpc_mean", "std": "validation_bpc_sd"})
curve_table.merge(bpc_means, on=["family", "fraction"]).to_csv(
    TABLE_DIR / "table_2_learning_curves_v211.csv", index=False)

sign_tests.to_csv(TABLE_DIR / "table_3_confirmatory_effects_v211.csv", index=False)
external_columns = [
    "validity",
    "unique_valid_yield",
    "strict_flp_yield",
    "novel_flp_yield",
    "final_candidate_yield",
    "mean_snn_to_train",
    "internal_diversity",
    "scaffold_novelty_vs_train",
]
external_table = (
    external_seeds.groupby(["model", "fraction"])[external_columns]
    .agg(["mean", "std"])
    .reset_index()
)
external_table.columns = [
    name if not detail else f"{name}_{detail}"
    for name, detail in external_table.columns
]
external_table.to_csv(
    TABLE_DIR / "table_4_external_generators_v211.csv", index=False)
review_models.drop(columns="order").to_csv(
    TABLE_DIR / "table_5_manual_validation_v211.csv", index=False)
examples.to_csv(TABLE_DIR / "representative_candidates_v211.csv", index=False)

manifest = {
    "evaluator_version": "2.1.1",
    "figures": sorted(path.name for path in FIGURE_DIR.iterdir()),
    "tables": sorted(path.name for path in TABLE_DIR.glob("*.csv")),
    "manual_validation": {
        "accepted": int(review_overall.accepted),
        "reviewed": int(review_overall.reviewed),
        "fraction": float(review_overall.acceptance_fraction),
    },
}
(TABLE_DIR / "publication_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

print("Tables:", TABLE_DIR)
print("Figures:", FIGURE_DIR)
