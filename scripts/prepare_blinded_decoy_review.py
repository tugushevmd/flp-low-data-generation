from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "blind_decoy_review"
OUTPUT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 20260805
rng = np.random.default_rng(RANDOM_SEED)

review = pd.read_csv(ROOT / "results" / "manual_validation" / "review_results.csv")
accepted = review[review["decision"] == "accept"].copy()

candidate_parts = []
for _, group in accepted.groupby(["model", "fraction"]):
    candidate_parts.append(group.sample(n=4, random_state=RANDOM_SEED))
candidates = pd.concat(candidate_parts, ignore_index=True)
candidates["panel_class"] = "model_candidate"
candidates["source_id"] = candidates["review_id"]
candidates["decoy_reason"] = ""

decoys = pd.read_csv(
    ROOT / "data" / "flp_curation" / "reference_smiles_excluded_v2.csv"
)
decoys = decoys[decoys["is_valid"]].copy()
decoys["panel_class"] = "decoy"
decoys["source_id"] = decoys["review_id"]
decoys["decoy_reason"] = decoys["review_reason"]
decoys["model"] = ""
decoys["fraction"] = np.nan

panel = pd.concat([
    candidates[[
        "canonical_smiles", "panel_class", "source_id",
        "decoy_reason", "model", "fraction",
    ]],
    decoys[[
        "canonical_smiles", "panel_class", "source_id",
        "decoy_reason", "model", "fraction",
    ]],
], ignore_index=True)
panel = panel.iloc[rng.permutation(len(panel))].reset_index(drop=True)
panel.insert(0, "blind_id", [f"B{i:03d}" for i in range(1, len(panel) + 1)])

review_sheet = panel[["blind_id", "canonical_smiles"]].copy()
review_sheet["decision"] = ""
review_sheet["comment"] = ""
review_sheet.to_csv(OUTPUT / "blind_review_sheet.csv", index=False)
panel.to_csv(OUTPUT / "blind_review_answer_key.csv", index=False)

for page, start in enumerate(range(0, len(panel), 8), start=1):
    page_data = panel.iloc[start:start + 8]
    molecules = [Chem.MolFromSmiles(smiles) for smiles in page_data["canonical_smiles"]]
    image = Draw.MolsToGridImage(
        molecules,
        molsPerRow=2,
        subImgSize=(600, 380),
        legends=page_data["blind_id"].tolist(),
        useSVG=False,
    )
    image.save(OUTPUT / f"blind_review_page_{page}.png")

summary = panel["panel_class"].value_counts().rename_axis("panel_class").reset_index(name="n")
summary.to_csv(OUTPUT / "blind_review_panel_summary.csv", index=False)

print(summary.to_string(index=False))
print("Review file:", OUTPUT / "blind_review_sheet.csv")
print("Keep the answer key closed until all decisions are recorded.")
