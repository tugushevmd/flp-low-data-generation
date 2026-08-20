from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures" / "candidate_selection"
OUTPUT.mkdir(parents=True, exist_ok=True)

table = pd.read_csv(ROOT / "results" / "manual_validation" / "review_results.csv")
table = table[table["decision"] == "accept"].copy()

models = ["GP-MoLFormer", "MolGPT", "REINVENT"]
training_sizes = {25: 42, 100: 166}
manifest = []

for fraction in [25, 100]:
    for model in models:
        group = table[
            (table["model"] == model) & (table["fraction"] == fraction)
        ].sort_values("review_id")

        molecules = [Chem.MolFromSmiles(smiles) for smiles in group["canonical_smiles"]]
        legends = group["review_id"].tolist()
        filename = f"{model.lower().replace('-', '_')}_{training_sizes[fraction]}_mol.png"

        image = Draw.MolsToGridImage(
            molecules,
            molsPerRow=3,
            subImgSize=(500, 360),
            legends=legends,
            useSVG=False,
        )
        image.save(OUTPUT / filename)

        for row in group.itertuples():
            manifest.append({
                "review_id": row.review_id,
                "model": model,
                "training_molecules": training_sizes[fraction],
                "canonical_smiles": row.canonical_smiles,
            })

pd.DataFrame(manifest).to_csv(OUTPUT / "candidate_manifest.csv", index=False)

print("Accepted candidates:", len(manifest))
print("Sheets:", OUTPUT)
