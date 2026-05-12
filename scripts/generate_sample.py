"""Generate PII-stripped sample CSV for instant demo.

Reads the full DataCo CSV, keeps only columns used by the pipeline,
samples 1000 rows, and writes to data/sample/sample.csv.

Usage: python scripts/generate_sample.py
"""

from pathlib import Path

import pandas as pd

# Columns the CSV adapter and pipeline actually use
KEEP_COLUMNS = [
    # Order-level
    "Order Id",
    "order date (DateOrders)",
    "Order Customer Id",
    "Order Region",
    "Order Country",
    "Order State",
    "Order Status",
    "Order City",
    "Order Zipcode",
    "Shipping Mode",
    "Days for shipment (scheduled)",
    # Financial
    "Benefit per order",
    "Sales per customer",
    "Order Profit Per Order",
    # Order items
    "Order Item Id",
    "Order Item Cardprod Id",
    "Order Item Quantity",
    "Order Item Product Price",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Profit Ratio",
    "Sales",
    "Order Item Total",
    # Product
    "Product Card Id",
    "Product Category Id",
    "Category Name",
    "Product Name",
    "Product Price",
    "Product Status",
    # Target
    "Late_delivery_risk",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
SAMPLE_PATH = SAMPLE_DIR / "sample.csv"


def main() -> None:
    if not RAW_PATH.exists():
        print(f"ERROR: Full dataset not found at {RAW_PATH}")
        print(
            "Download from: https://www.kaggle.com/datasets/"
            "shashwatwork/dataco-smart-supply-chain-for-big-data-analysis"
        )
        raise SystemExit(1)

    df = pd.read_csv(RAW_PATH, encoding="latin-1")

    # Keep only pipeline-relevant columns (strips PII: Customer Name, Email, etc.)
    available = [c for c in KEEP_COLUMNS if c in df.columns]
    df_clean = df[available]

    # Stratified sample: preserve late delivery rate distribution
    n_total = 1000
    late_rate = df_clean["Late_delivery_risk"].mean()
    n_late = round(n_total * late_rate)
    n_ontime = n_total - n_late

    late_rows = df_clean[df_clean["Late_delivery_risk"] == 1].sample(
        n=n_late, random_state=42
    )
    ontime_rows = df_clean[df_clean["Late_delivery_risk"] == 0].sample(
        n=n_ontime, random_state=42
    )
    sample = pd.concat([late_rows, ontime_rows]).sample(
        frac=1, random_state=42
    )

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_PATH, index=False)

    late_rate = sample["Late_delivery_risk"].mean()
    print(f"Sample generated: {len(sample)} rows, {len(available)} columns")
    print(f"Late delivery rate: {late_rate:.1%} (target: ~54.8%)")
    print(f"Saved to: {SAMPLE_PATH}")


if __name__ == "__main__":
    main()
