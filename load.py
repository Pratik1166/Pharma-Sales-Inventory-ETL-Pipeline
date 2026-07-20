"""
load.py — ETL Phase 3: Load
Loads cleaned pharma sales + inventory data into Postgres star schema.

Tables:
  dim_product    — drug & product attributes
  dim_region     — sales region master
  dim_date       — date dimension (pre-populated)
  dim_warehouse  — warehouse master
  fact_sales     — transactional sales fact
  fact_inventory — daily inventory snapshot
"""

import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, date
from sqlalchemy import create_engine, text

# ─────────────────────────────────────────────
# CONFIG — update with your Postgres credentials
# ─────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "pharma_dw",
    "user":     "etl_user",
    "password": "your_password",
}

DB_URL = (
    "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    .format(**DB_CONFIG)
)

PROCESSED_DIR  = Path("data/processed")
IF_EXISTS      = "append"   # "replace" for dev, "append" for prod

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("load.log")]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────
def get_engine():
    engine = create_engine(DB_URL, echo=False)
    log.info(f"Connected to Postgres → {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    return engine


# ─────────────────────────────────────────────
# DIMENSION LOADERS
# ─────────────────────────────────────────────
def load_dim_product(df: pd.DataFrame, engine) -> None:
    dim = df[[
        "product_id", "brand_name", "generic_name",
        "therapeutic_class", "manufacturer", "unit_cost",
        "schedule", "unit_of_measure"
    ]].drop_duplicates(subset="product_id")

    dim.to_sql("dim_product", engine, if_exists=IF_EXISTS, index=False,
               method="multi", chunksize=500)
    log.info(f"dim_product  : {len(dim):,} rows loaded ✓")


def load_dim_region(df: pd.DataFrame, engine) -> None:
    dim = df[["region"]].drop_duplicates()
    dim["region_id"] = range(1, len(dim) + 1)
    dim.rename(columns={"region": "region_name"}, inplace=True)

    dim.to_sql("dim_region", engine, if_exists=IF_EXISTS, index=False,
               method="multi")
    log.info(f"dim_region   : {len(dim):,} rows loaded ✓")


def load_dim_date(engine, start: str = "2022-01-01", end: str = "2025-12-31") -> None:
    dates = pd.date_range(start=start, end=end, freq="D")
    dim = pd.DataFrame({
        "date_id":    dates.strftime("%Y%m%d").astype(int),
        "date":       dates.date,
        "day":        dates.day,
        "week":       dates.isocalendar().week.astype(int),
        "month":      dates.month,
        "month_name": dates.strftime("%B"),
        "quarter":    dates.quarter,
        "year":       dates.year,
        "is_weekend": dates.weekday >= 5,
    })
    dim.to_sql("dim_date", engine, if_exists="replace", index=False,
               method="multi", chunksize=1000)
    log.info(f"dim_date     : {len(dim):,} rows loaded ✓")


def load_dim_warehouse(inv_df: pd.DataFrame, engine) -> None:
    dim = inv_df[[
        "warehouse_id", "temp_min_c", "temp_max_c", "cold_chain_required"
    ]].drop_duplicates(subset="warehouse_id")

    dim.to_sql("dim_warehouse", engine, if_exists=IF_EXISTS, index=False,
               method="multi")
    log.info(f"dim_warehouse: {len(dim):,} rows loaded ✓")


# ─────────────────────────────────────────────
# FACT LOADERS
# ─────────────────────────────────────────────
def load_fact_sales(df: pd.DataFrame, engine) -> None:
    fact = df[[
        "transaction_id", "rep_id", "product_id", "region",
        "sale_date", "units_sold", "unit_price", "discount_pct",
        "discount_amount", "revenue", "sale_month", "sale_year",
        "hospital_name"
    ]].copy()

    # Add surrogate date_id for joining with dim_date
    fact["date_id"] = pd.to_datetime(fact["sale_date"]).dt.strftime("%Y%m%d").astype(int)
    fact["loaded_at"] = datetime.utcnow()

    fact.to_sql("fact_sales", engine, if_exists=IF_EXISTS, index=False,
                method="multi", chunksize=5000)
    log.info(f"fact_sales   : {len(fact):,} rows loaded ✓")


def load_fact_inventory(inv_df: pd.DataFrame, engine) -> None:
    fact = inv_df[[
        "product_id", "warehouse_id", "batch_number",
        "quantity_on_hand", "reorder_threshold", "reorder_quantity",
        "avg_daily_units_sold", "days_of_stock_computed",
        "stockout_risk_computed", "cold_chain_required",
        "expiry_date", "last_updated", "extracted_at"
    ]].copy()

    fact["snapshot_date"] = date.today()
    fact["loaded_at"]     = datetime.utcnow()

    fact.to_sql("fact_inventory", engine, if_exists=IF_EXISTS, index=False,
                method="multi", chunksize=1000)
    log.info(f"fact_inventory: {len(fact):,} rows loaded ✓")


# ─────────────────────────────────────────────
# UPSERT HELPER (for incremental loads)
# ─────────────────────────────────────────────
def upsert_fact_sales(df: pd.DataFrame, engine) -> None:
    """
    Upsert fact_sales — insert new rows, skip existing transaction_ids.
    Uses Postgres INSERT ... ON CONFLICT DO NOTHING.
    """
    rows = df.to_dict(orient="records")
    upsert_sql = text("""
        INSERT INTO fact_sales (
            transaction_id, rep_id, product_id, region, sale_date,
            units_sold, unit_price, discount_pct, discount_amount,
            revenue, sale_month, sale_year, hospital_name, date_id, loaded_at
        ) VALUES (
            :transaction_id, :rep_id, :product_id, :region, :sale_date,
            :units_sold, :unit_price, :discount_pct, :discount_amount,
            :revenue, :sale_month, :sale_year, :hospital_name, :date_id, :loaded_at
        )
        ON CONFLICT (transaction_id) DO NOTHING
    """)
    with engine.begin() as conn:
        conn.execute(upsert_sql, rows)
    log.info(f"Upserted {len(rows):,} rows into fact_sales ✓")


# ─────────────────────────────────────────────
# LOAD QUALITY CHECK
# ─────────────────────────────────────────────
def post_load_check(engine) -> None:
    checks = {
        "fact_sales row count":        "SELECT COUNT(*) FROM fact_sales",
        "fact_inventory row count":    "SELECT COUNT(*) FROM fact_inventory",
        "dim_product row count":       "SELECT COUNT(*) FROM dim_product",
        "Stockout products today":     "SELECT COUNT(*) FROM fact_inventory WHERE stockout_risk_computed = TRUE AND snapshot_date = CURRENT_DATE",
        "Null product_id in sales":    "SELECT COUNT(*) FROM fact_sales WHERE product_id IS NULL",
    }
    log.info("\n── Post-load quality checks ──")
    with engine.connect() as conn:
        for label, query in checks.items():
            result = conn.execute(text(query)).scalar()
            log.info(f"  {label:<35}: {result:,}")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def load(sales_df: pd.DataFrame, inv_df: pd.DataFrame) -> None:
    log.info("── Load pipeline starting ──")
    engine = get_engine()

    # Dimensions first (FK dependencies)
    load_dim_date(engine)
    load_dim_product(sales_df, engine)
    load_dim_region(sales_df, engine)
    load_dim_warehouse(inv_df, engine)

    # Facts
    load_fact_sales(sales_df, engine)
    load_fact_inventory(inv_df, engine)

    # Quality gate
    post_load_check(engine)

    log.info("── Load pipeline complete ──")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Load latest processed files
    sales_files = sorted(PROCESSED_DIR.glob("pharma_sales_clean_*.csv"))
    inv_files   = sorted(PROCESSED_DIR.glob("inventory_clean_*.csv"))

    if not sales_files:
        raise FileNotFoundError("No cleaned sales CSV found. Run transform.py first.")
    if not inv_files:
        raise FileNotFoundError("No cleaned inventory CSV found. Run extract_inventory.py first.")

    sales_df = pd.read_csv(sales_files[-1])   # latest file
    inv_df   = pd.read_csv(inv_files[-1])

    log.info(f"Sales file    : {sales_files[-1].name} ({len(sales_df):,} rows)")
    log.info(f"Inventory file: {inv_files[-1].name}   ({len(inv_df):,} rows)")

    load(sales_df, inv_df)
