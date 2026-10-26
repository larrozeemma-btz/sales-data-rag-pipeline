"""Generate the synthetic beverage sales dataset used by the retrieval pipeline.

The schema mirrors the beverage sales dataset used in the original team assignment
(product, category, German region, customer type, quantity, discount, price), so the
retrieval behaviour can be reproduced and measured. Every row is fictional.

Usage:
    python data/generate_sales.py        # writes data/beverage_sales.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ORDERS = 10_000
OUT = Path(__file__).with_name("beverage_sales.csv")

# Region names are kept in German, as in the original dataset.
REGIONS = [
    "Baden-Württemberg", "Bayern", "Berlin", "Brandenburg", "Bremen", "Hamburg",
    "Hessen", "Mecklenburg-Vorpommern", "Niedersachsen", "Nordrhein-Westfalen",
    "Rheinland-Pfalz", "Saarland", "Sachsen", "Sachsen-Anhalt", "Schleswig-Holstein", "Thüringen",
]
REGION_WEIGHTS = np.array([11, 13, 4.5, 3, 1, 2.2, 7.5, 2, 9.5, 21.5, 4.9, 1.2, 4.9, 2.6, 3.5, 2.6])

# (product, category, unit price in EUR)
PRODUCTS = [
    ("Coca-Cola", "Soft Drinks", 1.49), ("Pepsi", "Soft Drinks", 1.39), ("Fanta", "Soft Drinks", 1.29),
    ("Sprite", "Soft Drinks", 1.29), ("Mezzo Mix", "Soft Drinks", 1.35), ("Club-Mate", "Soft Drinks", 1.65),
    ("Schweppes Tonic", "Soft Drinks", 1.55), ("Red Bull", "Energy Drinks", 1.95), ("Monster Energy", "Energy Drinks", 1.85),
    ("Rockstar", "Energy Drinks", 1.45), ("Gerolsteiner", "Water", 0.79), ("Volvic", "Water", 0.95),
    ("Evian", "Water", 1.15), ("Vittel", "Water", 0.89), ("Apollinaris", "Water", 0.99),
    ("Hohes C Orange", "Juices", 2.49), ("Granini Apple", "Juices", 2.29), ("Rauch Multivitamin", "Juices", 2.39),
    ("Innocent Smoothie", "Juices", 3.29), ("Krombacher Pils", "Alcoholic Beverages", 1.19),
    ("Beck's", "Alcoholic Beverages", 1.25), ("Paulaner Weissbier", "Alcoholic Beverages", 1.49),
    ("Erdinger Weissbier", "Alcoholic Beverages", 1.45), ("Jägermeister", "Alcoholic Beverages", 14.99),
    ("Moët Brut", "Alcoholic Beverages", 39.90), ("Tchibo Coffee Beans", "Hot Drinks", 7.99),
    ("Jacobs Krönung", "Hot Drinks", 6.49), ("Teekanne Green Tea", "Hot Drinks", 2.79),
]

CUSTOMER_TYPES = ["B2B", "B2C"]
B2B_DISCOUNTS = [0, 5, 10, 15]


def generate(n_orders=N_ORDERS, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    customers = {"B2B": [f"CUS{i:04d}" for i in range(1, 801)], "B2C": [f"CUS{i:04d}" for i in range(801, 6001)]}
    dates = pd.date_range("2021-01-01", "2023-12-31", freq="D")
    for o in range(1, n_orders + 1):
        ctype = rng.choice(CUSTOMER_TYPES, p=[0.35, 0.65])
        region = rng.choice(REGIONS, p=REGION_WEIGHTS / REGION_WEIGHTS.sum())
        cust = rng.choice(customers[ctype])
        date = dates[rng.integers(len(dates))].date().isoformat()
        n_lines = rng.integers(1, 6)
        for idx in rng.choice(len(PRODUCTS), size=n_lines, replace=False):
            product, category, price = PRODUCTS[idx]
            if ctype == "B2B":
                qty = int(rng.integers(10, 120))
                discount = int(rng.choice(B2B_DISCOUNTS, p=[0.4, 0.25, 0.2, 0.15]))
            else:
                qty = int(rng.integers(1, 13))
                discount = 0
            unit = round(price * rng.uniform(0.95, 1.05), 2)
            total = round(qty * unit * (1 - discount / 100), 2)
            rows.append({
                "Order_ID": f"ORD{o:05d}", "Customer_ID": cust, "Customer_Type": ctype,
                "Product": product, "Category": category, "Region": region, "Order_Date": date,
                "Unit_Price": unit, "Quantity": qty, "Discount": discount, "Total_Price": total,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} rows to {OUT}")
