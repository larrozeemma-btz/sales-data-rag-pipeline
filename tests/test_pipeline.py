"""Run with:  python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.pipeline import HybridRetriever, Index, TfidfRetriever, answer  # noqa: E402

INDEX = Index(pd.read_csv(ROOT / "data" / "beverage_sales.csv"))


class ParserTests(unittest.TestCase):
    def test_english_region_maps_to_german_name(self):
        self.assertEqual(INDEX.parse("Coca-Cola sold in Bavaria").region, "Bayern")

    def test_longest_region_name_wins(self):
        self.assertEqual(INDEX.parse("Fanta in Sachsen-Anhalt").region, "Sachsen-Anhalt")
        self.assertEqual(INDEX.parse("Fanta in Lower Saxony").region, "Niedersachsen")
        self.assertEqual(INDEX.parse("Fanta in Saxony").region, "Sachsen")

    def test_discount_and_customer_type(self):
        c = INDEX.parse("Water bought by business customers with a 15% discount")
        self.assertEqual((c.category, c.customer_type, c.discount), ("Water", "B2B", 15))


class RetrieverTests(unittest.TestCase):
    def test_hybrid_rows_satisfy_every_constraint(self):
        rows = HybridRetriever(INDEX).retrieve("Show Coca-Cola sold in Bavaria with a 10% discount")
        self.assertEqual(len(rows), 5)
        self.assertTrue((rows.Product == "Coca-Cola").all())
        self.assertTrue((rows.Region == "Bayern").all())
        self.assertTrue((rows.Discount == 10).all())

    def test_tfidf_baseline_misses_region_and_discount(self):
        rows = TfidfRetriever(INDEX).retrieve("Show Coca-Cola sold in Bavaria with a 10% discount")
        self.assertFalse(((rows.Region == "Bayern") & (rows.Discount == 10)).all())

    def test_impossible_question_returns_no_rows(self):
        rows = HybridRetriever(INDEX).retrieve("Coca-Cola sold to B2C customers with a 15% discount")
        self.assertTrue(rows.empty)
        self.assertIn("No records match", answer("q", rows))


if __name__ == "__main__":
    unittest.main()
