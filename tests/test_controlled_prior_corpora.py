import json
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "controlled_priors"


class ControlledPriorCorpusTests(unittest.TestCase):
    def test_corpus_design(self):
        summary = pd.read_csv(DATA_DIR / "corpus_summary.csv")
        expected_doses = {
            "P0": 0.0,
            "P0.1": 0.001,
            "P1": 0.01,
            "P5": 0.05,
        }

        self.assertEqual(set(summary["prior"]), set(expected_doses))
        self.assertTrue((summary["molecules"] == 150000).all())
        self.assertTrue((summary["unique_smiles"] == 150000).all())

        for prior, expected in expected_doses.items():
            observed = summary.loc[
                summary["prior"] == prior,
                "main_group_fraction",
            ].iloc[0]
            self.assertAlmostEqual(observed, expected)

    def test_leakage_and_matching(self):
        leakage = json.loads(
            (DATA_DIR / "leakage_report.json").read_text(encoding="utf-8")
        )
        balance = pd.read_csv(DATA_DIR / "matching_balance.csv")

        self.assertEqual(leakage["selected_flp_like_molecules"], 0)
        self.assertEqual(leakage["exact_matches_to_curated"], 0)
        self.assertEqual(leakage["exact_matches_to_templates"], 0)
        self.assertLess(leakage["max_snn_to_curated"], 0.4)
        self.assertLess(
            balance["standardized_mean_difference"].abs().max(),
            0.2,
        )


if __name__ == "__main__":
    unittest.main()
