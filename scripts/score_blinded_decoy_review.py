from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = ROOT / "results" / "blind_decoy_review"
TABLE_DIR = ROOT / "results" / "publication_tables"

REJECTED = {
    "B001", "B002", "B004", "B010", "B012", "B021", "B022",
    "B026", "B029", "B035", "B037", "B039", "B044",
}

OUT_OF_SCOPE = {"B007", "B034", "B042"}

sheet = pd.read_csv(REVIEW_DIR / "blind_review_sheet.csv")
sheet["decision"] = sheet["blind_id"].map(
    lambda blind_id: "reject" if blind_id in REJECTED else "accept"
)
sheet.to_csv(REVIEW_DIR / "blind_review_sheet.csv", index=False)

key = pd.read_csv(REVIEW_DIR / "blind_review_answer_key.csv")
scored = sheet.merge(key.drop(columns="canonical_smiles"), on="blind_id")
scored["true_candidate"] = scored["panel_class"] == "model_candidate"
scored["accepted"] = scored["decision"] == "accept"
scored["scope_status"] = scored["blind_id"].map(
    lambda blind_id: "out_of_scope_non_boron_LA"
    if blind_id in OUT_OF_SCOPE else "within_B_centered_scope"
)

tp = int((scored["true_candidate"] & scored["accepted"]).sum())
fn = int((scored["true_candidate"] & ~scored["accepted"]).sum())
tn = int((~scored["true_candidate"] & ~scored["accepted"]).sum())
fp = int((~scored["true_candidate"] & scored["accepted"]).sum())


def metric_row(metric, successes, total):
    interval = binomtest(successes, total).proportion_ci(method="wilson")
    return {
        "metric": metric,
        "successes": successes,
        "total": total,
        "value": successes / total,
        "ci95_low": interval.low,
        "ci95_high": interval.high,
    }


summary = pd.DataFrame([
    metric_row("candidate acceptance rate", tp, tp + fn),
    metric_row("decoy rejection rate", tn, tn + fp),
    metric_row("agreement with panel labels", tp + tn, len(scored)),
    metric_row("accepted-structure precision", tp, tp + fp),
])

within_scope_decoys = scored[
    (~scored["true_candidate"])
    & (scored["scope_status"] == "within_B_centered_scope")
]
within_scope_rejected = int((~within_scope_decoys["accepted"]).sum())
within_scope_accepted = scored[
    scored["accepted"]
    & (scored["scope_status"] == "within_B_centered_scope")
]
summary = pd.concat([summary, pd.DataFrame([
    metric_row(
        "B-centered decoy rejection rate",
        within_scope_rejected,
        len(within_scope_decoys),
    ),
    metric_row(
        "B-centered accepted-structure precision",
        int(within_scope_accepted["true_candidate"].sum()),
        len(within_scope_accepted),
    ),
])], ignore_index=True)

confusion = pd.DataFrame([{
    "true_candidate_accepted": tp,
    "true_candidate_rejected": fn,
    "decoy_rejected": tn,
    "decoy_accepted": fp,
}])

scored.to_csv(REVIEW_DIR / "blind_review_scored.csv", index=False)
summary.to_csv(REVIEW_DIR / "blind_review_metrics.csv", index=False)
summary.to_csv(TABLE_DIR / "table_s11_blind_decoy_review.csv", index=False)
confusion.to_csv(REVIEW_DIR / "blind_review_confusion_matrix.csv", index=False)

mistakes = scored[scored["true_candidate"] != scored["accepted"]]
mistakes.to_csv(REVIEW_DIR / "blind_review_misclassifications.csv", index=False)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))

plot_metrics = summary.iloc[:3]
x = np.arange(len(plot_metrics))
values = plot_metrics["value"].to_numpy()
lower = values - plot_metrics["ci95_low"].to_numpy()
upper = plot_metrics["ci95_high"].to_numpy() - values
axes[0].errorbar(
    x, values, yerr=[lower, upper], fmt="o", color="#8A3F50",
    ecolor="#8A3F50", markersize=8, linewidth=2, capsize=4,
)
axes[0].set_xticks(x, ["Candidate\nacceptance", "Decoy\nrejection", "Label\nagreement"])
axes[0].set_ylim(0, 1.08)
axes[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
axes[0].set_title("Blinded review performance")
axes[0].grid(axis="y", color="#E7E8EA", linewidth=0.8)
axes[0].text(-0.13, 1.06, "A", transform=axes[0].transAxes,
             fontsize=13, fontweight="bold", va="top")

decoy_counts = (
    scored[~scored["true_candidate"]]
    .groupby(["decoy_reason", "decision"])
    .size()
    .unstack(fill_value=0)
)
reason_labels = {
    "no_LA": "No recognised LA",
    "no_LB": "No recognised LB",
    "no_LA_or_LB": "No LA or LB",
    "direct_B_N_bond": "Direct B-N bond",
    "direct_B_N_bond, direct_B_P_bond": "Direct B-N/P bonds",
    "phosphine_oxide_or_sulfide": "P(V) oxide/sulfide",
}
decoy_counts = decoy_counts.reindex(reason_labels).fillna(0)
labels = [reason_labels[reason] for reason in decoy_counts.index]
y = np.arange(len(decoy_counts))
rejected = decoy_counts.get("reject", pd.Series(0, index=decoy_counts.index))
accepted_decoys = decoy_counts.get("accept", pd.Series(0, index=decoy_counts.index))
axes[1].barh(y, rejected, color="#356B52", label="rejected")
axes[1].barh(y, accepted_decoys, left=rejected, color="#8A3F50", label="accepted")
axes[1].set_yticks(y, labels)
axes[1].invert_yaxis()
axes[1].set_xlabel("Structures")
axes[1].set_title("Outcome by decoy class")
axes[1].legend(ncol=2, loc="lower right")
axes[1].grid(axis="x", color="#E7E8EA", linewidth=0.8)
axes[1].text(-0.13, 1.06, "B", transform=axes[1].transAxes,
             fontsize=13, fontweight="bold", va="top")

fig.suptitle("Candidate-plus-decoy chemical review",
             fontsize=14, fontweight="semibold")
fig.tight_layout()
for suffix in ["png", "pdf"]:
    fig.savefig(ROOT / "figures" / f"figure_s4_blind_decoy_review.{suffix}",
                dpi=320 if suffix == "png" else None, bbox_inches="tight")
plt.close(fig)

print(confusion.to_string(index=False))
print()
print(summary.round(4).to_string(index=False))
print()
print("Misclassified structures:", ", ".join(mistakes["blind_id"]))
