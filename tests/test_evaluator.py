import csv
from pathlib import Path
import unittest

from evaluation import (
    audit_generation,
    canonicalize,
    chemical_sanity,
    classify_flp,
    inspect_smiles,
)


ROOT = Path(__file__).resolve().parents[1]


class EvaluatorTests(unittest.TestCase):
    def test_control_table(self):
        with (ROOT / "evaluation" / "controls.csv").open(
            encoding="utf-8",
            newline="",
        ) as handle:
            controls = list(csv.DictReader(handle))

        for control in controls:
            with self.subTest(control=control["id"]):
                inspected = inspect_smiles(control["smiles"])
                classification = classify_flp(
                    inspected["canonical_smiles"]
                )
                sanity = chemical_sanity(
                    inspected["canonical_smiles"]
                )
                strict = (
                    inspected["is_valid"]
                    and sanity["chemically_sane"]
                    and sanity["is_neutral"]
                    and classification["has_LA"]
                    and classification["has_LB"]
                    and not classification["negative_flags"]
                )

                self.assertEqual(
                    inspected["is_valid"],
                    bool(int(control["expected_valid"])),
                )
                self.assertEqual(
                    classification["has_LA"],
                    bool(int(control["expected_has_LA"])),
                )
                self.assertEqual(
                    classification["has_LB"],
                    bool(int(control["expected_has_LB"])),
                )
                self.assertEqual(
                    strict,
                    bool(int(control["expected_strict"])),
                )

    def test_primary_identity_ignores_stereochemistry(self):
        left = "F[C@H](Cl)Br"
        right = "F[C@@H](Cl)Br"
        self.assertEqual(canonicalize(left), canonicalize(right))
        self.assertNotEqual(
            canonicalize(left, isomeric=True),
            canonicalize(right, isomeric=True),
        )

    def test_direct_bond_is_detected_from_graph(self):
        result = classify_flp(canonicalize("B(C)(C)P(C)C"))
        self.assertIn("direct_B_P_bond", result["negative_flags"])

    def test_undercoordinated_boron_boron_bond_is_blocked(self):
        result = classify_flp(canonicalize("CN(C)c1ccccc1BB(F)Cl"))
        self.assertTrue(result["has_LA"])
        self.assertTrue(result["has_LB"])
        self.assertIn("undercoordinated_B_B_bond", result["negative_flags"])

    def test_tricoordinate_boron_boron_bond_is_allowed(self):
        result = classify_flp(canonicalize("B(C)(C)B(C)CCP(C)C"))
        self.assertNotIn("undercoordinated_B_B_bond", result["negative_flags"])

    def test_p_o_substituted_phosphorus_is_not_a_phosphine(self):
        result = classify_flp(
            canonicalize("B(c1ccccc1)(c1ccccc1)c1ccccc1CCP(OC)OC")
        )
        self.assertNotIn("LB_phosphine", result["LB_types"])
        self.assertIn("LB_other_P", result["excluded_LB_types"])

    def test_explicit_hydrogen_does_not_change_identity(self):
        self.assertEqual(
            canonicalize("c1ccccc1"),
            canonicalize("[H]c1ccccc1"),
        )

    def test_failure_taxonomy(self):
        branch = inspect_smiles("CC(C")
        ring = inspect_smiles("C1CC")
        valence = inspect_smiles("C(C)(C)(C)(C)C")

        self.assertIn(
            "unbalanced_parentheses",
            branch["failure_reasons"],
        )
        self.assertIn(
            "unmatched_ring_closure",
            ring["failure_reasons"],
        )
        self.assertIn("valence_problem", valence["failure_reasons"])

    def test_scaffold_error_does_not_stop_evaluation(self):
        cesium_smiles = (
            "Cl[132Cs]c1c(c2c(Cl)c(Cl)c(Cl)c(Cl)c2Cl)"
            "CC1(C(C)CC)CB(c2c(Cl)c(Cl)c(Cl)c(Cl)c2Cl)CC1"
        )
        rows, _, _ = audit_generation(
            raw_smiles=[cesium_smiles],
            train_smiles=["c1ccccc1"],
            seed_smiles=["c1ccccc1"],
            template_smiles=[],
        )

        self.assertFalse(rows.iloc[0]["chemically_sane"])

    def test_invalid_smiles_does_not_stop_evaluation(self):
        rows, summary, _ = audit_generation(
            raw_smiles=["C(C", "c1ccccc1"],
            train_smiles=["c1ccccc1"],
            seed_smiles=["c1ccccc1"],
            template_smiles=[],
        )

        self.assertFalse(rows.iloc[0]["is_valid"])
        self.assertFalse(rows.iloc[0]["chemically_sane"])
        self.assertEqual(summary.iloc[0]["raw_n"], 2)


if __name__ == "__main__":
    unittest.main()
