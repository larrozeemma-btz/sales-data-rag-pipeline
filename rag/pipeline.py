"""Retrieval-augmented question answering over tabular sales data.

Two retrievers share the same index:

* TfidfRetriever   reproduces the original team approach: every row becomes a short text
                   document, embedded with TF-IDF, and ranked by cosine similarity.
* HybridRetriever  first reads hard constraints out of the question (product, category,
                   region, customer type, discount), filters rows on them, then ranks the
                   remaining rows with the same TF-IDF similarity.

`answer()` turns retrieved rows into a grounded summary that names exactly what was found,
so a reader can see when the evidence does not match the question.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# English and alternative spellings mapped to the German region names used in the data.
REGION_ALIASES = {
    "bavaria": "Bayern", "saxony-anhalt": "Sachsen-Anhalt", "saxony anhalt": "Sachsen-Anhalt",
    "lower saxony": "Niedersachsen", "saxony": "Sachsen", "north rhine-westphalia": "Nordrhein-Westfalen",
    "north rhine westphalia": "Nordrhein-Westfalen", "nrw": "Nordrhein-Westfalen", "hesse": "Hessen",
    "thuringia": "Thüringen", "thueringen": "Thüringen", "rhineland-palatinate": "Rheinland-Pfalz",
    "rhineland palatinate": "Rheinland-Pfalz", "mecklenburg-western pomerania": "Mecklenburg-Vorpommern",
    "mecklenburg western pomerania": "Mecklenburg-Vorpommern", "baden-wuerttemberg": "Baden-Württemberg",
    "baden wurttemberg": "Baden-Württemberg",
}
CUSTOMER_ALIASES = {"b2b": "B2B", "business": "B2B", "wholesale": "B2B", "b2c": "B2C", "consumer": "B2C", "retail": "B2C"}


def row_to_document(row: pd.Series) -> str:
    """Turn one sales row into the text that gets embedded."""
    return (
        f"Product: {row.Product}. Category: {row.Category}. Region: {row.Region}. "
        f"Customer type: {row.Customer_Type}. Discount: {row.Discount}%. "
        f"Quantity: {row.Quantity}. Total price: {row.Total_Price:.2f} EUR."
    )


@dataclass
class Constraints:
    product: str | None = None
    category: str | None = None
    region: str | None = None
    customer_type: str | None = None
    discount: int | None = None
    unmatched: list[str] = field(default_factory=list)

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != "unmatched" and v is not None}


class Index:
    """TF-IDF index over the sales table, plus the vocabulary used to parse questions."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.documents = self.df.apply(row_to_document, axis=1).tolist()
        self.vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b[\w-]+\b")
        self.matrix = self.vectorizer.fit_transform(self.documents)
        self._vocab = {
            "product": sorted(df.Product.unique(), key=len, reverse=True),
            "category": sorted(df.Category.unique(), key=len, reverse=True),
            "region": sorted(df.Region.unique(), key=len, reverse=True),
        }

    def similarity(self, query: str) -> np.ndarray:
        return linear_kernel(self.vectorizer.transform([query]), self.matrix).ravel()

    def parse(self, question: str) -> Constraints:
        """Read hard constraints out of a question. Longest names are matched first so that
        'Sachsen-Anhalt' is not read as 'Sachsen'."""
        q = question.lower()
        c = Constraints()

        def take(names):
            nonlocal q
            for name in names:
                pattern = r"(?<![\w-])" + re.escape(name.lower()) + r"(?![\w-])"
                if re.search(pattern, q):
                    q = re.sub(pattern, " ", q)
                    return name
            return None

        for alias in sorted(REGION_ALIASES, key=len, reverse=True):
            pattern = r"(?<![\w-])" + re.escape(alias) + r"(?![\w-])"
            if re.search(pattern, q):
                c.region = REGION_ALIASES[alias]
                q = re.sub(pattern, " ", q)
                break
        c.region = c.region or take(self._vocab["region"])
        c.product = take(self._vocab["product"])
        c.category = take(self._vocab["category"])
        for alias, value in CUSTOMER_ALIASES.items():
            if re.search(rf"\b{alias}\b", q):
                c.customer_type = value
                break
        m = re.search(r"(\d{1,2})\s*%", q)
        if m:
            c.discount = int(m.group(1))
        return c


class TfidfRetriever:
    name = "TF-IDF only"

    def __init__(self, index: Index):
        self.index = index

    def retrieve(self, question: str, k: int = 5) -> pd.DataFrame:
        scores = self.index.similarity(question)
        top = np.argsort(-scores, kind="stable")[:k]
        return self.index.df.iloc[top].assign(similarity=scores[top].round(3))


class HybridRetriever:
    name = "Constraint filter + TF-IDF"

    def __init__(self, index: Index):
        self.index = index

    def retrieve(self, question: str, k: int = 5) -> pd.DataFrame:
        c = self.index.parse(question)
        df = self.index.df
        mask = np.ones(len(df), dtype=bool)
        column = {"product": "Product", "category": "Category", "region": "Region",
                  "customer_type": "Customer_Type", "discount": "Discount"}
        for key, value in c.as_dict().items():
            mask &= (df[column[key]] == value).to_numpy()
        scores = self.index.similarity(question)
        if not mask.any():
            # Nothing satisfies every constraint: return nothing rather than near misses.
            return df.iloc[[]].assign(similarity=[])
        scores = np.where(mask, scores, -1.0)
        top = np.argsort(-scores, kind="stable")[: min(k, int(mask.sum()))]
        return df.iloc[top].assign(similarity=scores[top].round(3))


def answer(question: str, rows: pd.DataFrame) -> str:
    """Grounded summary built only from the retrieved rows."""
    if rows.empty:
        return (f"No records match every condition in: \"{question}\". "
                "Try removing a condition (region, discount, or customer type).")
    def found(col):
        return ", ".join(sorted(rows[col].astype(str).unique()))
    return (
        f"Based on the {len(rows)} most relevant records: "
        f"products {found('Product')}; regions {found('Region')}; "
        f"customer types {found('Customer_Type')}; discounts {found('Discount')}%. "
        f"Average quantity {rows.Quantity.mean():.1f} units, average order line "
        f"{rows.Total_Price.mean():.2f} EUR (range {rows.Total_Price.min():.2f} to {rows.Total_Price.max():.2f} EUR)."
    )
