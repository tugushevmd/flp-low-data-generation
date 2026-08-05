from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "publication_tables"
FIGURES = ROOT / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Character 5-gram": "#A8ADB3",
    "Controlled GRU P5": "#555A61",
    "GP-MoLFormer": "#356B52",
    "MolGPT": "#8A3F50",
    "REINVENT": "#75658C",
}


def summarize_runs(table, model, method, fraction):
    data = table[table["fraction"] == fraction]
    return {
        "model": model,
        "method": method,
        "flp_train_n": {25: 42, 100: 166}[fraction],
        "runs": len(data),
        "validity_mean": data["validity"].mean(),
        "validity_sd": data["validity"].std(),
        "strict_flp_yield_mean": data["strict_flp_yield"].mean(),
        "strict_flp_yield_sd": data["strict_flp_yield"].std(),
        "final_candidate_yield_mean": data["final_candidate_yield"].mean(),
        "final_candidate_yield_sd": data["final_candidate_yield"].std(),
        "comparison_scope": "same evaluator; architectures and pretraining differ",
    }


rows = []

controlled = pd.read_csv(
    RESULTS / "controlled_prior_learning_curves" / "training_seed_summary_v2.csv"
)
controlled = controlled[controlled["family"] == "P5"]
for fraction in [25, 100]:
    rows.append(summarize_runs(
        controlled, "Controlled GRU P5", "controlled-prior GRU", fraction
    ))

for model, folder in [
    ("GP-MoLFormer", "external_gpmolformer"),
    ("MolGPT", "external_molgpt"),
    ("REINVENT", "external_reinvent"),
]:
    table = pd.read_csv(RESULTS / folder / "training_seed_summary_v2.csv")
    for fraction in sorted(table["fraction"].unique()):
        rows.append(summarize_runs(table, model, "pretrained generator", fraction))

char_runs = pd.read_csv(RESULTS / "baselines" / "char_5gram" / "run_summary.csv")
rows.append({
    "model": "Character 5-gram",
    "method": "data-matched non-neural baseline",
    "flp_train_n": 166,
    "runs": len(char_runs),
    "validity_mean": char_runs["validity"].mean(),
    "validity_sd": char_runs["validity"].std(),
    "strict_flp_yield_mean": char_runs["strict_flp_yield"].mean(),
    "strict_flp_yield_sd": char_runs["strict_flp_yield"].std(),
    "final_candidate_yield_mean": char_runs["final_candidate_yield"].mean(),
    "final_candidate_yield_sd": char_runs["final_candidate_yield"].std(),
    "comparison_scope": "same 166 FLP molecules and same evaluator",
})

model_table = pd.DataFrame(rows).sort_values(["flp_train_n", "model"])
model_table.to_csv(TABLES / "table_s7_model_context.csv", index=False)

fragment = pd.read_csv(
    RESULTS / "baselines" / "fragment_recombination" / "aggregate_summary.csv"
).set_index("metric")
fragment_table = pd.DataFrame([{
    "method": "Fragment recombination enumerator",
    "library_size": 3780,
    "validity_mean": fragment.loc["validity", "mean"],
    "strict_flp_yield_mean": fragment.loc["strict_flp_yield", "mean"],
    "template_relative_final_yield": fragment.loc["final_candidate_yield", "mean"],
    "fragment_candidate_yield": fragment.loc["fragment_candidate_yield", "mean"],
    "interpretation": (
        "rule-based chemical upper bound; template-relative novelty is zero by construction"
    ),
}])
fragment_table.to_csv(TABLES / "table_s8_fragment_baseline.csv", index=False)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), sharey=True)
for ax, train_n, letter in zip(axes, [42, 166], "AB"):
    data = model_table[model_table["flp_train_n"] == train_n].copy()
    order = [name for name in COLORS if name in set(data["model"])]
    data = data.set_index("model").loc[order].reset_index()
    x = np.arange(len(data))
    values = data["final_candidate_yield_mean"].to_numpy()
    errors = data["final_candidate_yield_sd"].fillna(0).to_numpy()
    ax.bar(x, values, yerr=errors, capsize=3,
           color=[COLORS[name] for name in data["model"]], width=0.68)
    for position, value in zip(x, values):
        ax.text(position, value + max(errors.max(), 0.003) + 0.003,
                f"{value:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, data["model"], rotation=25, ha="right")
    ax.set_title(f"{train_n} FLP molecules")
    ax.set_ylabel("Final-candidate yield" if train_n == 42 else "")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", color="#E7E8EA", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")

fig.suptitle("Model context under the frozen evaluator",
             fontsize=14, fontweight="semibold")
fig.tight_layout()
fig.savefig(FIGURES / "figure_s3_model_context.png", dpi=320, bbox_inches="tight")
fig.savefig(FIGURES / "figure_s3_model_context.pdf", bbox_inches="tight")
plt.close(fig)

print(model_table[[
    "model", "flp_train_n", "validity_mean",
    "strict_flp_yield_mean", "final_candidate_yield_mean",
]].round(4).to_string(index=False))
