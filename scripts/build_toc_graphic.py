from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
COLORS = {
    "green": "#356B52",
    "wine": "#8A3F50",
    "violet": "#75658C",
    "graphite": "#555A61",
    "light": "#E7E8EA",
}

candidate = pd.read_csv(
    ROOT / "results" / "publication_tables" / "representative_candidates_v211.csv"
)
smiles = candidate.loc[candidate["review_id"] == "E030", "canonical_smiles"].iloc[0]
mol = Chem.MolFromSmiles(smiles)
mol_image = Draw.MolToImage(mol, size=(520, 400), kekulize=True)

fig = plt.figure(figsize=(9.75, 5.25), dpi=100, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(0.04, 0.91, "PRETRAINING CHEMISTRY", fontsize=18, weight="bold", color="#202225")
ax.text(0.04, 0.84, "changes low-data FLP-like generation", fontsize=13, color=COLORS["graphite"])

for y, label, color, filled in [
    (0.65, "P0   generic prior", COLORS["graphite"], 0),
    (0.42, "P5   5% main-group prior", COLORS["wine"], 1),
]:
    ax.text(0.05, y + 0.08, label, fontsize=12, weight="bold", color=color)
    for i in range(20):
        x = 0.055 + (i % 10) * 0.024
        yy = y - (i // 10) * 0.055
        dot_color = COLORS["green"] if i < filled else COLORS["light"]
        ax.scatter(x, yy, s=95, color=dot_color, edgecolor="white", linewidth=0.5)

box = FancyBboxPatch((0.35, 0.38), 0.20, 0.26, boxstyle="round,pad=0.015,rounding_size=0.01",
                     linewidth=1.2, edgecolor=COLORS["graphite"], facecolor="#F7F7F8")
ax.add_patch(box)
ax.text(0.45, 0.56, "same GRU", ha="center", fontsize=14, weight="bold")
ax.text(0.45, 0.49, "+ 166 FLP structures", ha="center", fontsize=11)
ax.text(0.45, 0.43, "fixed evaluator", ha="center", fontsize=11, color=COLORS["graphite"])
ax.add_patch(FancyArrowPatch((0.29, 0.53), (0.35, 0.53), arrowstyle="-|>", mutation_scale=15,
                             linewidth=1.5, color=COLORS["graphite"]))
ax.add_patch(FancyArrowPatch((0.55, 0.53), (0.61, 0.53), arrowstyle="-|>", mutation_scale=15,
                             linewidth=1.5, color=COLORS["graphite"]))

image_ax = fig.add_axes([0.61, 0.25, 0.35, 0.62])
image_ax.imshow(mol_image)
image_ax.axis("off")

ax.text(0.66, 0.17, "strict FLP-like yield", fontsize=11, color=COLORS["graphite"])
ax.text(0.66, 0.08, "8.4%", fontsize=21, weight="bold", color=COLORS["graphite"])
ax.text(0.765, 0.085, "→", fontsize=22, color=COLORS["green"])
ax.text(0.82, 0.08, "12.6%", fontsize=21, weight="bold", color=COLORS["wine"])

fig.savefig(FIGURES / "toc_graphic.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
fig.savefig(FIGURES / "toc_graphic.svg", bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
