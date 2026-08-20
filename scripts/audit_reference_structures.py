from pathlib import Path
import json

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "flp_curation" / "reference_smiles_curated_v2.csv"
OUTPUT = ROOT / "results" / "publication_tables" / "table_s20_reference_structure_audit.csv"
SUMMARY = ROOT / "results" / "publication_tables" / "reference_structure_audit_summary.json"


def audit_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"parse_ok": False}

    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    roundtrip = Chem.MolFromSmiles(canonical)
    elements = sorted({atom.GetSymbol() for atom in mol.GetAtoms()})

    return {
        "parse_ok": True,
        "canonical_roundtrip_ok": roundtrip is not None,
        "canonical_audit": canonical,
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 4),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "components": len(Chem.GetMolFrags(mol)),
        "formal_charge": Chem.GetFormalCharge(mol),
        "radical_electrons": sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()),
        "stereocenters": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "elements": ";".join(elements),
        "B": sum(atom.GetSymbol() == "B" for atom in mol.GetAtoms()),
        "P": sum(atom.GetSymbol() == "P" for atom in mol.GetAtoms()),
        "N": sum(atom.GetSymbol() == "N" for atom in mol.GetAtoms()),
        "Al": sum(atom.GetSymbol() == "Al" for atom in mol.GetAtoms()),
        "Si": sum(atom.GetSymbol() == "Si" for atom in mol.GetAtoms()),
    }


table = pd.read_csv(INPUT)
audit = pd.DataFrame([audit_smiles(smiles) for smiles in table["canonical_smiles"]])
result = pd.concat([table.reset_index(names="reference_index"), audit], axis=1)
result["canonical_match"] = result["canonical_smiles"] == result["canonical_audit"]
result.to_csv(OUTPUT, index=False)

summary = {
    "records": len(result),
    "parse_ok": int(result["parse_ok"].sum()),
    "canonical_roundtrip_ok": int(result["canonical_roundtrip_ok"].sum()),
    "canonical_match": int(result["canonical_match"].sum()),
    "single_component": int((result["components"] == 1).sum()),
    "neutral": int((result["formal_charge"] == 0).sum()),
    "without_radicals": int((result["radical_electrons"] == 0).sum()),
}
SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(pd.Series(summary).to_string())
