from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "controlled_priors"
CURATION_DIR = ROOT / "data" / "flp_curation"


class LearningCurveSubsetTests(unittest.TestCase):
    def test_nested_subsets(self):
        table = pd.read_csv(DATA_DIR / "learning_curve_subsets.csv")

        self.assertEqual(len(table), 166)
        self.assertEqual(table["smiles"].nunique(), 166)
        self.assertEqual(table["in_25_percent"].sum(), 42)
        self.assertEqual(table["in_50_percent"].sum(), 83)
        self.assertEqual(table["in_100_percent"].sum(), 166)

        self.assertTrue(
            (
                ~table["in_25_percent"]
                | table["in_50_percent"]
            ).all()
        )
        self.assertTrue(
            (
                ~table["in_50_percent"]
                | table["in_100_percent"]
            ).all()
        )

    def test_full_subset_matches_curated_train(self):
        table = pd.read_csv(DATA_DIR / "learning_curve_subsets.csv")
        curated = pd.read_csv(CURATION_DIR / "train_smiles.csv")

        self.assertEqual(set(table["smiles"]), set(curated["smiles"]))

    def test_small_subset_is_scaffold_diverse(self):
        table = pd.read_csv(DATA_DIR / "learning_curve_subsets.csv")

        self.assertEqual(table.iloc[:42]["scaffold"].nunique(), 42)
        self.assertEqual(table.iloc[:83]["scaffold"].nunique(), 70)


if __name__ == "__main__":
    unittest.main()
