"""Measure retrieval quality for both retrievers on a labelled question set.

Each test question is generated from a template together with its ground-truth
constraints, so a retrieved row is relevant only if it satisfies every constraint.

Metric: precision@5 = share of the top 5 rows that satisfy every constraint in the question.

Usage:
    python eval/evaluate.py      # writes eval/results.csv, docs/evaluation.md, docs/precision_by_query_type.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag.pipeline import HybridRetriever, Index, TfidfRetriever  # noqa: E402

K = 5
PER_TEMPLATE = 10
SEED = 7
ENGLISH_NAMES = {
    "Bayern": "Bavaria", "Sachsen": "Saxony", "Niedersachsen": "Lower Saxony", "Hessen": "Hesse",
    "Thüringen": "Thuringia", "Nordrhein-Westfalen": "North Rhine-Westphalia", "Rheinland-Pfalz": "Rhineland-Palatinate",
}
NICKNAMES = {
    "Coca-Cola": "Coke", "Beck's": "Becks", "Jägermeister": "Jager", "Moët Brut": "Moet champagne",
    "Hohes C Orange": "orange juice", "Red Bull": "Redbull", "Club-Mate": "Mate", "Paulaner Weissbier": "Paulaner wheat beer",
}
COLUMN = {"product": "Product", "category": "Category", "region": "Region", "customer_type": "Customer_Type", "discount": "Discount"}

TEMPLATES = [
    ("Product + region (German name)", "{product} orders from {region}", ["product", "region"]),
    ("Category + customer type", "Find {category} purchased by {customer_type} customers", ["category", "customer_type"]),
    ("Product + region (German name) + discount", "Show {product} sold in {region} with a {discount}% discount", ["product", "region", "discount"]),
    ("Product + region (English name) + discount", "Show {product} sold in {region_en} with a {discount}% discount", ["product", "region", "discount"]),
    ("Category + region (English name) + customer type", "{category} sold to {customer_type} customers in {region_en}", ["category", "region", "customer_type"]),
    ("Product nickname + region (English name) + discount", "Show {product_nick} sold in {region_en} with a {discount}% discount", ["product", "region", "discount"]),
]


def build_questions(df, rng):
    questions = []
    for qtype, template, keys in TEMPLATES:
        needs_english = "{region_en}" in template
        combos = df[[COLUMN[k] for k in keys]].drop_duplicates()
        if needs_english:
            combos = combos[combos.Region.isin(ENGLISH_NAMES)]
        if "{product_nick}" in template:
            combos = combos[combos.Product.isin(NICKNAMES)]
        if "discount" in keys:
            combos = combos[combos.Discount > 0]
        counts = df.groupby([COLUMN[k] for k in keys]).size()
        combos = combos[[counts[tuple(r)] >= K for r in combos.itertuples(index=False)]]
        for r in combos.sample(PER_TEMPLATE, random_state=int(rng.integers(1e6))).itertuples(index=False):
            truth = dict(zip(keys, r))
            fill = {**truth, "region_en": ENGLISH_NAMES.get(truth.get("region"), ""),
                    "product_nick": NICKNAMES.get(truth.get("product"), "")}
            questions.append({"query_type": qtype, "question": template.format(**fill), "truth": truth})
    return questions


def precision_at_k(rows, truth):
    if rows.empty:
        return 0.0
    ok = np.ones(len(rows), dtype=bool)
    for key, value in truth.items():
        ok &= (rows[COLUMN[key]] == value).to_numpy()
    return ok.sum() / K


def main():
    df = pd.read_csv(ROOT / "data" / "beverage_sales.csv")
    index = Index(df)
    retrievers = [TfidfRetriever(index), HybridRetriever(index)]
    questions = build_questions(df, np.random.default_rng(SEED))

    records = []
    for q in questions:
        for r in retrievers:
            rows = r.retrieve(q["question"], k=K)
            records.append({"query_type": q["query_type"], "question": q["question"], "retriever": r.name,
                            "precision_at_5": precision_at_k(rows, q["truth"])})
    results = pd.DataFrame(records)
    results.to_csv(ROOT / "eval" / "results.csv", index=False)

    order = [t[0] for t in TEMPLATES]
    table = (results.pivot_table(index="query_type", columns="retriever", values="precision_at_5", aggfunc="mean")
             .reindex(order)[[r.name for r in retrievers]])
    overall = results.groupby("retriever").precision_at_5.mean()[[r.name for r in retrievers]]

    lines = ["# Retrieval evaluation", "",
             f"{len(questions)} labelled questions ({PER_TEMPLATE} per type), precision@{K}: the share of the top {K} "
             "rows that satisfy every condition in the question.", "",
             "| Question type | " + " | ".join(table.columns) + " |", "|---|" + "---|" * len(table.columns)]
    for qtype, row in table.iterrows():
        lines.append(f"| {qtype} | " + " | ".join(f"{v:.0%}" for v in row) + " |")
    lines.append("| **All questions** | " + " | ".join(f"**{v:.0%}**" for v in overall) + " |")
    (ROOT / "docs" / "evaluation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(table)


def plot(table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.6, 6.6), dpi=100)
    y = np.arange(len(table))
    h = 0.36
    colors = {"TF-IDF only": "#B9C2D3", "Constraint filter + TF-IDF": "#2E5EAA"}
    for i, col in enumerate(table.columns):
        vals = table[col].to_numpy()
        bars = ax.barh(y + (i - 0.5) * h, vals, height=h, color=colors[col], label=col)
        for b, v in zip(bars, vals):
            ax.text(v + 0.012, b.get_y() + b.get_height() / 2, f"{v:.0%}", va="center", fontsize=10, color="#2B3245")
    ax.set_yticks(y)
    ax.set_yticklabels([t.replace(" + ", "\n+ ", 1) if len(t) > 32 else t for t in table.index], fontsize=10, color="#2B3245")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color="#5B6478")
    ax.set_xlabel("Precision@5: share of top 5 rows matching every condition", color="#5B6478")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#D9DDE5")
    ax.spines["bottom"].set_color("#D9DDE5")
    ax.set_title("Retrieval precision by question type", loc="left", fontsize=14, fontweight="bold", color="#1F3864", pad=14)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "precision_by_query_type.png")


if __name__ == "__main__":
    main()
