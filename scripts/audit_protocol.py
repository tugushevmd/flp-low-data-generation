from pathlib import Path
import hashlib
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABLES = ROOT / "results" / "publication_tables"


def read_smiles(path):
    table = pd.read_csv(path)
    return set(table.iloc[:, 0].dropna().astype(str))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


split_dir = DATA / "flp_curation"
split_manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
train = read_smiles(split_dir / "train_smiles.csv")
validation = read_smiles(split_dir / "validation_smiles.csv")
reference = read_smiles(split_dir / "reference_smiles.csv")

prior_dir = DATA / "controlled_priors"
leakage = json.loads((prior_dir / "leakage_report.json").read_text(encoding="utf-8"))
corpus_manifest = json.loads((prior_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
balance = pd.read_csv(prior_dir / "matching_balance.csv")
composition = pd.read_csv(TABLES / "table_s2_composition_differences.csv")

effects = pd.read_csv(TABLES / "table_s4_discovery_confirmatory_effects.csv")
selected = effects[
    (effects["cohort"] == "pooled")
    & (effects["fraction"] == 100)
    & (effects["metric"] == "validation_bpc")
].iloc[0]
fixed = effects[
    (effects["cohort"] == "pooled")
    & (effects["fraction"] == 100)
    & (effects["metric"] == "validation_bpc_fixed_8000")
].iloc[0]

rows = [
    ("FLP split", "train molecules", len(train), split_manifest["train_molecules"]),
    ("FLP split", "validation molecules", len(validation), split_manifest["validation_molecules"]),
    ("FLP split", "exact train-validation overlap", len(train & validation), 0),
    ("FLP split", "train contained in full reference", train <= reference, True),
    ("FLP split", "validation contained in full reference", validation <= reference, True),
    ("Pretraining leakage", "selected main-group molecules", leakage["selected_main_group_molecules"], None),
    ("Pretraining leakage", "exact matches to curated FLP", leakage["exact_matches_to_curated"], 0),
    ("Pretraining leakage", "exact matches to templates", leakage["exact_matches_to_templates"], 0),
    ("Pretraining leakage", "maximum SNN to curated FLP", leakage["max_snn_to_curated"], corpus_manifest["max_snn_to_curated_flp"]),
    ("Pretraining leakage", "mean SNN to curated FLP", leakage["mean_snn_to_curated"], None),
    ("Corpus balance", "maximum matched-component SMD", balance["standardized_mean_difference"].abs().max(), 0.2),
    ("Corpus balance", "maximum full-corpus SMD versus P0", composition["standardized_mean_difference"].abs().max(), 0.2),
    ("Checkpoint robustness", "P5-P0 BPC improvement, selected checkpoint", selected["mean_improvement"], None),
    ("Checkpoint robustness", "P5-P0 BPC improvement, fixed 8000 exposures", fixed["mean_improvement"], None),
]

audit = pd.DataFrame(rows, columns=["section", "check", "value", "target_or_limit"])
audit.to_csv(TABLES / "table_s9_protocol_audit.csv", index=False)

hash_rows = []
for filename, expected in split_manifest["files"].items():
    observed = sha256(split_dir / filename)
    hash_rows.append({
        "file": filename,
        "expected_sha256": expected,
        "observed_sha256": observed,
        "matches_manifest": observed == expected,
    })
pd.DataFrame(hash_rows).to_csv(TABLES / "table_s10_file_integrity.csv", index=False)

print(audit.to_string(index=False))
print("\nAll split-file hashes match:", all(row["matches_manifest"] for row in hash_rows))
