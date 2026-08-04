from collections import Counter
import re

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


EVALUATOR_VERSION = "2.1.1"
RDKIT_VERSION = rdBase.rdkitVersion

SNN_MIN = 0.40
SNN_MAX = 0.90
NEAR_DUPLICATE = 0.95

ALLOWED_ATOMS = {
    "H", "B", "C", "N", "O", "F", "P", "S",
    "Cl", "Br", "I", "Si", "Al", "Ge",
}

RDLogger.DisableLog("rdApp.*")
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def ring_tokens_are_balanced(smiles):
    outside_brackets = []
    in_brackets = False
    for character in smiles:
        if character == "[":
            in_brackets = True
        elif character == "]":
            in_brackets = False
        elif not in_brackets:
            outside_brackets.append(character)

    tokens = re.findall(r"%\d{2}|\d", "".join(outside_brackets))
    return all(count % 2 == 0 for count in Counter(tokens).values())


def inspect_smiles(smiles, reached_max_length=False):
    text = "" if pd.isna(smiles) else str(smiles).strip()
    reasons = []

    if reached_max_length:
        reasons.append("reached_max_length")
    if not text:
        return {
            "raw_smiles": text,
            "is_valid": False,
            "canonical_smiles": None,
            "isomeric_smiles": None,
            "failure_reasons": reasons + ["empty_output"],
        }
    if text.count("(") != text.count(")"):
        reasons.append("unbalanced_parentheses")
    if not ring_tokens_are_balanced(text):
        reasons.append("unmatched_ring_closure")

    mol = Chem.MolFromSmiles(text, sanitize=False)
    if mol is None:
        return {
            "raw_smiles": text,
            "is_valid": False,
            "canonical_smiles": None,
            "isomeric_smiles": None,
            "failure_reasons": reasons + ["parse_failure"],
        }

    for problem in Chem.DetectChemistryProblems(mol):
        problem_name = type(problem).__name__
        if "Valence" in problem_name:
            reasons.append("valence_problem")
        elif "Kekulize" in problem_name:
            reasons.append("kekulization_failure")
        else:
            reasons.append(problem_name)

    sanitize_result = Chem.SanitizeMol(mol, catchErrors=True)
    is_valid = sanitize_result == Chem.SanitizeFlags.SANITIZE_NONE
    if not is_valid:
        reasons.append("sanitization_failure")

    canonical = None
    isomeric = None
    if is_valid:
        mol = Chem.RemoveHs(mol)
        canonical = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=False,
        )
        isomeric = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )

    return {
        "raw_smiles": text,
        "is_valid": is_valid,
        "canonical_smiles": canonical,
        "isomeric_smiles": isomeric,
        "failure_reasons": sorted(set(reasons)),
    }


def canonicalize(smiles, isomeric=False):
    result = inspect_smiles(smiles)
    column = "isomeric_smiles" if isomeric else "canonical_smiles"
    return result[column]


def chemical_sanity(smiles):
    if smiles is None:
        return {
            "chemically_sane": False,
            "sanity_reasons": ["invalid_smiles"],
            "formal_charge": np.nan,
            "heavy_atoms": np.nan,
            "is_neutral": False,
            "is_single_component": False,
        }

    mol = Chem.MolFromSmiles(smiles)
    reasons = []
    atoms = {atom.GetSymbol() for atom in mol.GetAtoms()}
    charge = Chem.GetFormalCharge(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    components = len(Chem.GetMolFrags(mol))

    if atoms - ALLOWED_ATOMS:
        reasons.append("unsupported_atom")
    if abs(charge) > 1:
        reasons.append("forbidden_charge")
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        reasons.append("radical")
    if components > 1:
        reasons.append("multiple_components")
    if heavy_atoms < 10:
        reasons.append("too_small")
    if heavy_atoms > 120:
        reasons.append("too_large")
    if any(atom.HasValenceViolation() for atom in mol.GetAtoms()):
        reasons.append("valence_problem")

    return {
        "chemically_sane": not reasons,
        "sanity_reasons": sorted(set(reasons)),
        "formal_charge": charge,
        "heavy_atoms": heavy_atoms,
        "is_neutral": charge == 0,
        "is_single_component": components == 1,
    }


def aryl_ring_halogens(mol, atom_index):
    counts = []
    for ring in mol.GetRingInfo().AtomRings():
        if atom_index not in ring:
            continue
        count = 0
        for index in ring:
            atom = mol.GetAtomWithIdx(index)
            for neighbor in atom.GetNeighbors():
                if (
                    neighbor.GetIdx() not in ring
                    and neighbor.GetSymbol() in {"F", "Cl", "Br", "I"}
                ):
                    count += 1
        counts.append(count)
    return max(counts, default=0)


def is_amide_nitrogen(atom):
    for neighbor in atom.GetNeighbors():
        if neighbor.GetSymbol() != "C":
            continue
        for bond in neighbor.GetBonds():
            other = bond.GetOtherAtom(neighbor)
            if (
                other.GetIdx() != atom.GetIdx()
                and other.GetSymbol() in {"O", "S"}
                and bond.GetBondType() == Chem.BondType.DOUBLE
            ):
                return True
    return False


def is_nitro_nitrogen(atom):
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() <= 0:
        return False
    oxygen_neighbors = [
        neighbor for neighbor in atom.GetNeighbors()
        if neighbor.GetSymbol() == "O"
    ]
    return len(oxygen_neighbors) >= 2


def has_p_double_bond_to_chalcogen(atom):
    return any(
        bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetOtherAtom(atom).GetSymbol() in {"O", "S"}
        for bond in atom.GetBonds()
    )


def classify_flp(smiles):
    empty_result = {
        "has_LA": False,
        "has_LB": False,
        "LA_types": [],
        "LB_types": [],
        "excluded_LB_types": [],
        "negative_flags": [],
        "flp_tier": "invalid",
        "min_LA_LB_bonds": np.nan,
    }
    if smiles is None:
        return empty_result

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return empty_result

    la_atoms = []
    lb_atoms = []
    la_types = []
    lb_types = []
    excluded_lb_types = []
    negative_flags = []

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        degree = atom.GetDegree()

        if symbol == "B":
            if charge < 0:
                negative_flags.append("borate")
            if charge == 0 and degree == 3:
                la_atoms.append(atom.GetIdx())
                la_types.append("LA_B_tricoordinate")
                aryl_neighbors = [
                    neighbor
                    for neighbor in atom.GetNeighbors()
                    if neighbor.GetSymbol() == "C" and neighbor.GetIsAromatic()
                ]
                if aryl_neighbors:
                    la_types.append("LA_B_aryl")
                if len(aryl_neighbors) >= 2:
                    la_types.append("LA_B_diaryl")
                if any(
                    aryl_ring_halogens(mol, neighbor.GetIdx()) >= 2
                    for neighbor in aryl_neighbors
                ):
                    la_types.append("LA_B_polyhalogenated_aryl")

        if symbol == "P":
            if charge > 0:
                negative_flags.append("phosphonium")
                excluded_lb_types.append("LB_phosphonium")
                continue
            if has_p_double_bond_to_chalcogen(atom):
                negative_flags.append("phosphine_oxide_or_sulfide")
                excluded_lb_types.append("LB_P_chalcogenide")
                continue

            heavy_neighbors = [
                neighbor.GetSymbol()
                for neighbor in atom.GetNeighbors()
                if neighbor.GetSymbol() != "H"
            ]
            carbon_or_boron_substituents = (
                heavy_neighbors
                and all(symbol in {"C", "B"} for symbol in heavy_neighbors)
            )
            only_single_bonds = all(
                bond.GetBondType() == Chem.BondType.SINGLE
                for bond in atom.GetBonds()
            )
            if (
                charge == 0
                and degree in {1, 2, 3}
                and carbon_or_boron_substituents
                and only_single_bonds
            ):
                lb_atoms.append(atom.GetIdx())
                lb_types.append("LB_phosphine")
            elif charge == 0:
                excluded_lb_types.append("LB_other_P")

        if symbol == "N":
            if is_nitro_nitrogen(atom):
                excluded_lb_types.append("LB_nitro_N")
                continue
            if charge > 0 and degree == 4:
                excluded_lb_types.append("LB_quaternary_ammonium")
                continue
            if is_amide_nitrogen(atom):
                excluded_lb_types.append("LB_amide_N")
                continue
            if atom.GetIsAromatic() and atom.GetTotalNumHs() > 0:
                excluded_lb_types.append("LB_pyrrolic_N")
                continue

            all_single_bonds = all(
                bond.GetBondType() == Chem.BondType.SINGLE
                for bond in atom.GetBonds()
            )
            is_amine = not atom.GetIsAromatic() and all_single_bonds
            is_aromatic_base = (
                atom.GetIsAromatic()
                and atom.GetTotalNumHs() == 0
            )
            if charge == 0 and (is_amine or is_aromatic_base):
                lb_atoms.append(atom.GetIdx())
                lb_types.append(
                    "LB_aromatic_N" if is_aromatic_base else "LB_amine"
                )

    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "B":
            continue
        for neighbor in atom.GetNeighbors():
            neighbor_symbol = neighbor.GetSymbol()
            if neighbor_symbol == "B" and (
                atom.GetDegree() < 3
                or neighbor.GetDegree() < 3
                or atom.GetTotalNumHs() > 0
                or neighbor.GetTotalNumHs() > 0
            ):
                negative_flags.append("undercoordinated_B_B_bond")
            if neighbor_symbol in {"P", "N"}:
                negative_flags.append(
                    f"direct_B_{neighbor_symbol}_bond"
                )

    distances = [
        len(Chem.GetShortestPath(mol, la_index, lb_index)) - 1
        for la_index in la_atoms
        for lb_index in lb_atoms
    ]

    has_la = bool(la_atoms)
    has_lb = bool(lb_atoms)
    blocking_flags = sorted(set(negative_flags))
    if has_la and has_lb and not blocking_flags:
        tier = "LA_LB_no_negative_flags"
    elif has_la and has_lb:
        tier = "LA_LB_with_negative_flags"
    elif has_la:
        tier = "LA_only"
    elif has_lb:
        tier = "LB_only"
    else:
        tier = "no_LA_LB"

    return {
        "has_LA": has_la,
        "has_LB": has_lb,
        "LA_types": sorted(set(la_types)),
        "LB_types": sorted(set(lb_types)),
        "excluded_LB_types": sorted(set(excluded_lb_types)),
        "negative_flags": blocking_flags,
        "flp_tier": tier,
        "min_LA_LB_bonds": min(distances) if distances else np.nan,
    }


def fingerprints(smiles_values):
    return [
        FPGEN.GetFingerprint(Chem.MolFromSmiles(smiles))
        for smiles in smiles_values
    ]


def nearest_similarities(smiles_values, reference_smiles):
    if not smiles_values or not reference_smiles:
        return []
    reference_fps = fingerprints(reference_smiles)
    return [
        max(DataStructs.BulkTanimotoSimilarity(fp, reference_fps))
        for fp in fingerprints(smiles_values)
    ]


def internal_diversity(smiles_values):
    fps = fingerprints(smiles_values)
    similarities = []
    for index, fp in enumerate(fps[:-1]):
        similarities.extend(
            DataStructs.BulkTanimotoSimilarity(fp, fps[index + 1:])
        )
    return 1 - np.mean(similarities) if similarities else np.nan


def murcko_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except (RuntimeError, ValueError):
        return None
    if scaffold.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(
        scaffold,
        canonical=True,
        isomericSmiles=False,
    )


def canonical_set(smiles_values, isomeric=False):
    values = {
        canonicalize(smiles, isomeric=isomeric)
        for smiles in smiles_values
    }
    return {value for value in values if value is not None}


def audit_generation(
    raw_smiles,
    train_smiles,
    seed_smiles,
    template_smiles,
    reached_max_length=None,
):
    raw_smiles = list(raw_smiles)
    if reached_max_length is None:
        reached_max_length = [False] * len(raw_smiles)

    inspected = [
        inspect_smiles(smiles, reached_limit)
        for smiles, reached_limit in zip(raw_smiles, reached_max_length)
    ]
    rows = pd.DataFrame(inspected)
    rows["is_unique_valid"] = (
        rows["is_valid"]
        & ~rows["canonical_smiles"].duplicated()
    )

    valid_counts = (
        rows.loc[rows["is_valid"], "canonical_smiles"]
        .value_counts()
    )
    rows["generation_count"] = (
        rows["canonical_smiles"]
        .map(valid_counts)
        .fillna(0)
        .astype(int)
    )

    train_set = canonical_set(train_smiles)
    seed_set = canonical_set(seed_smiles)
    template_set = canonical_set(template_smiles)
    all_reference = seed_set | template_set

    train_isomeric = canonical_set(train_smiles, isomeric=True)
    seed_isomeric = canonical_set(seed_smiles, isomeric=True)
    template_isomeric = canonical_set(template_smiles, isomeric=True)
    all_reference_isomeric = seed_isomeric | template_isomeric

    unique_smiles = rows.loc[
        rows["is_unique_valid"], "canonical_smiles"
    ].tolist()
    similarity = dict(zip(
        unique_smiles,
        nearest_similarities(unique_smiles, sorted(train_set)),
    ))
    rows["snn_to_train"] = rows["canonical_smiles"].map(similarity)

    rows["novel_vs_train"] = ~rows["canonical_smiles"].isin(train_set)
    rows["novel_vs_all_seed"] = ~rows["canonical_smiles"].isin(seed_set)
    rows["novel_vs_templates"] = ~rows["canonical_smiles"].isin(template_set)
    rows["novel_vs_all_reference"] = (
        ~rows["canonical_smiles"].isin(all_reference)
    )
    rows["stereo_novel_vs_train"] = (
        ~rows["isomeric_smiles"].isin(train_isomeric)
    )
    rows["stereo_novel_vs_all_reference"] = (
        ~rows["isomeric_smiles"].isin(all_reference_isomeric)
    )

    sanity = rows["canonical_smiles"].map(chemical_sanity).apply(pd.Series)
    classes = rows["canonical_smiles"].map(classify_flp).apply(pd.Series)
    rows = pd.concat([rows, sanity, classes], axis=1)

    boolean_columns = [
        "chemically_sane",
        "is_neutral",
        "is_single_component",
        "has_LA",
        "has_LB",
    ]
    for column in boolean_columns:
        rows[column] = rows[column].fillna(False).astype(bool)

    rows["no_negative_flags"] = rows["negative_flags"].map(
        lambda value: isinstance(value, list) and not value
    )
    rows["is_strict_flp_like"] = (
        rows["is_unique_valid"]
        & rows["chemically_sane"]
        & rows["is_neutral"]
        & rows["has_LA"]
        & rows["has_LB"]
        & rows["no_negative_flags"]
    )
    rows["is_novel_flp_like"] = (
        rows["is_strict_flp_like"]
        & rows["novel_vs_all_reference"]
    )
    rows["is_final_candidate"] = (
        rows["is_novel_flp_like"]
        & rows["snn_to_train"].between(SNN_MIN, SNN_MAX)
        & (rows["snn_to_train"] < NEAR_DUPLICATE)
    )

    unique = rows[rows["is_unique_valid"]]
    strict = rows[rows["is_strict_flp_like"]]
    novel_strict = rows[rows["is_novel_flp_like"]]

    train_scaffolds = {
        scaffold
        for scaffold in map(murcko_smiles, train_set)
        if scaffold
    }
    generated_scaffolds = [
        scaffold
        for scaffold in map(murcko_smiles, unique["canonical_smiles"])
        if scaffold
    ]

    summary = {
        "evaluator_version": EVALUATOR_VERSION,
        "raw_n": len(rows),
        "validity": rows["is_valid"].mean(),
        "unique_valid_n": len(unique),
        "unique_valid_yield": rows["is_unique_valid"].mean(),
        "chemically_sane_fraction": unique["chemically_sane"].mean(),
        "neutral_fraction": unique["is_neutral"].mean(),
        "strict_flp_n": len(strict),
        "strict_flp_yield": rows["is_strict_flp_like"].mean(),
        "novel_flp_n": len(novel_strict),
        "novel_flp_yield": rows["is_novel_flp_like"].mean(),
        "final_candidate_n": int(rows["is_final_candidate"].sum()),
        "final_candidate_yield": rows["is_final_candidate"].mean(),
        "novelty_vs_all_seed": unique["novel_vs_all_seed"].mean(),
        "novelty_vs_templates": unique["novel_vs_templates"].mean(),
        "novelty_vs_all_reference": (
            unique["novel_vs_all_reference"].mean()
        ),
        "stereo_novelty_vs_all_reference": (
            unique["stereo_novel_vs_all_reference"].mean()
        ),
        "near_duplicate_095": (
            unique["snn_to_train"] >= NEAR_DUPLICATE
        ).mean(),
        "mean_snn_to_train": unique["snn_to_train"].mean(),
        "internal_diversity": internal_diversity(
            unique["canonical_smiles"].tolist()
        ),
        "scaffold_novelty_vs_train": (
            np.mean([
                scaffold not in train_scaffolds
                for scaffold in generated_scaffolds
            ])
            if generated_scaffolds
            else np.nan
        ),
    }

    funnel = pd.DataFrame({
        "stage": [
            "raw",
            "valid",
            "unique valid",
            "chemically sane",
            "neutral",
            "strict FLP-like",
            "novel FLP-like",
            "final candidates",
        ],
        "n": [
            len(rows),
            int(rows["is_valid"].sum()),
            int(rows["is_unique_valid"].sum()),
            int((rows["is_unique_valid"] & rows["chemically_sane"]).sum()),
            int((rows["is_unique_valid"] & rows["is_neutral"]).sum()),
            int(rows["is_strict_flp_like"].sum()),
            int(rows["is_novel_flp_like"].sum()),
            int(rows["is_final_candidate"].sum()),
        ],
    })
    funnel["fraction_of_raw"] = funnel["n"] / len(rows)
    return rows, pd.DataFrame([summary]), funnel
