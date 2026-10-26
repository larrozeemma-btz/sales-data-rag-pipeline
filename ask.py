"""Ask a question of the sales data and compare both retrievers.

Usage:
    python ask.py "Show Coca-Cola sold in Bavaria with a 10% discount"
"""
import sys
from pathlib import Path

import pandas as pd

from rag.pipeline import HybridRetriever, Index, TfidfRetriever, answer

DATA = Path(__file__).parent / "data" / "beverage_sales.csv"
COLUMNS = ["Product", "Region", "Customer_Type", "Discount", "Quantity", "Total_Price", "similarity"]


def main(question: str):
    index = Index(pd.read_csv(DATA))
    print(f"Question: {question}")
    print(f"Constraints read from the question: {index.parse(question).as_dict()}\n")
    for retriever in (TfidfRetriever(index), HybridRetriever(index)):
        rows = retriever.retrieve(question)
        print(f"== {retriever.name}")
        print(rows[COLUMNS].to_string(index=False) if not rows.empty else "(no rows)")
        print(answer(question, rows), "\n")


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]) or "Show Coca-Cola sold in Bavaria with a 10% discount")
