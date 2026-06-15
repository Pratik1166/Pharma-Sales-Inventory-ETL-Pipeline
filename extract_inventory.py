"""
extract_inventory.py — ETL Phase 1b: Extract Inventory from REST API
Paginates through the Pharma Inventory API and extracts all stock records.

Features:
  - Full pagination via next_page_url
  - Incremental load using last_updated timestamp
  - Retry logic with exponential backoff
  - Cold-chain field parsing (storage_temp_c → temp_min_c, temp_max_c)
  - Stockout risk recomputation for verification
  - Saves raw JSON + processed CSV
"""

import requests
import pandas as pd
import json
import time
import logging
import re
from pathlib import Path
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_URL        = "https://api.pharma-warehouse.internal/api/v2/inventory/stock"
PAGE_SIZE       = 100
MAX_RETRIES     = 3
BACKOFF_BASE    = 2            # exponential backoff: 2s, 4s, 8s
REQUEST_TIMEOUT = 15
STOCKOUT_DAYS   = 14           # industry standard threshold

OUTPUT_DIR      = Path("data/raw")
PROCESSED_DIR   = Path("data/processed")
STATE_FILE      = Path("data/.last_inventory_run")   # stores last run timestamp

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("extract_inventory.log")
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# INCREMENTAL LOAD — track last run
# ─────────────────────────────────────────────
def get_last_run_timestamp() -> str | None:
    """Read the timestamp from the last successful run (for incremental loads)."""
    if STATE_FILE.exists():
        ts = STATE_FILE.read_text().strip()
        log.info(f"Incremental mode — fetching records updated since: {ts}")
        return ts
    log.info("No state file found — performing full extract")
    return None


def save_run_timestamp() -> None:
    """Persist current UTC timestamp after a successful run."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    STATE_FILE.write_text(now)
    log.info(f"Run timestamp saved → {now}")


# ─────────────────────────────────────────────
# HTTP — single page fetch with retry + backoff
# ─────────────────────────────────────────────
def fetch_page(url: str, params: dict) -> dict | None:
    """
    GET one page. Retries with exponential backoff on failure.
    Returns parsed JSON or None after all retries exhausted.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"  GET {url} params={params} (attempt {attempt})")
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            if status == 429:
                wait = int(e.response.headers.get("Retry-After", 60))
                log.warning(f"  Rate limited — waiting {wait}s before retry")
                time.sleep(wait)
            else:
                log.warning(f"  HTTP {status}: {e}")

        except requests.exceptions.ConnectionError:
            log.warning(f"  Connection error (attempt {attempt})")

        except requests.exceptions.Timeout:
            log.warning(f"  Timeout after {REQUEST_TIMEOUT}s (attempt {attempt})")

        wait = BACKOFF_BASE ** attempt
        log.info(f"  Retrying in {wait}s...")
        time.sleep(wait)

    log.error(f"  All {MAX_RETRIES} attempts failed for {url}")
    return None


# ─────────────────────────────────────────────
# PARSE — flatten one API record
# ─────────────────────────────────────────────
def parse_temp(temp_str: str | None) -> tuple[float | None, float | None]:
    """
    Parse storage_temp_c string into numeric min/max.
    Handles: "2–8", "15-25", "2 to 8", "below 25"
    """
    if not temp_str:
        return None, None

    temp_str = str(temp_str).replace("–", "-").replace(" to ", "-")

    # Range: "2-8" or "15-25"
    match = re.match(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", temp_str)
    if match:
        return float(match.group(1)), float(match.group(2))

    # Single bound: "below 25"
    match = re.match(r"(?:below|under|<)\s*(-?\d+(?:\.\d+)?)", temp_str, re.IGNORECASE)
    if match:
        return None, float(match.group(1))

    return None, None


def parse_record(r: dict) -> dict:
    """Flatten and enrich a single inventory record."""
    temp_min, temp_max = parse_temp(r.get("storage_temp_c"))

    qty   = r.get("quantity_on_hand", 0) or 0
    avg   = r.get("avg_daily_units_sold") or 0

    # Recompute days_of_stock ourselves — don't trust API value blindly
    days_of_stock_computed = round(qty / avg, 1) if avg > 0 else None

    # Recompute stockout_risk using our own threshold
    stockout_risk_computed = (
        days_of_stock_computed is not None and days_of_stock_computed < STOCKOUT_DAYS
    )

    return {
        # Identifiers
        "product_id":               r.get("product_id"),
        "warehouse_id":             r.get("warehouse_id"),
        "batch_number":             r.get("batch_number"),

        # Product info
        "brand_name":               r.get("brand_name"),
        "generic_name":             r.get("generic_name"),
        "unit_of_measure":          r.get("unit_of_measure"),

        # Stock levels
        "quantity_on_hand":         qty,
        "reorder_threshold":        r.get("reorder_threshold"),
        "reorder_quantity":         r.get("reorder_quantity"),
        "avg_daily_units_sold":     avg,

        # KPIs — API provided
        "days_of_stock_api":        r.get("days_of_stock"),
        "stockout_risk_api":        r.get("stockout_risk"),

        # KPIs — recomputed by us for verification
        "days_of_stock_computed":   days_of_stock_computed,
        "stockout_risk_computed":   stockout_risk_computed,

        # Mismatch flag — alert if API and our calc disagree
        "kpi_mismatch":             (
            r.get("days_of_stock") != days_of_stock_computed
        ),

        # Cold chain (parsed from string)
        "storage_temp_c_raw":       r.get("storage_temp_c"),
        "temp_min_c":               temp_min,
        "temp_max_c":               temp_max,
        "cold_chain_required":      temp_max is not None and temp_max <= 8,

        # Dates
        "expiry_date":              r.get("expiry_date"),
        "last_updated":             r.get("last_updated"),

        # Audit
        "extracted_at":             datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ─────────────────────────────────────────────
# MAIN EXTRACT
# ─────────────────────────────────────────────
def extract_inventory(incremental: bool = True) -> pd.DataFrame:
    """
    Paginate the inventory API and return a clean DataFrame.

    Args:
        incremental: If True, only fetch records updated since last run.
                     If False, perform a full extract.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_raw     = []
    all_records = []
    page_num    = 1
    next_url    = BASE_URL

    # Build initial query params
    params = {"page_size": PAGE_SIZE}
    if incremental:
        last_run = get_last_run_timestamp()
        if last_run:
            params["updated_since"] = last_run

    log.info(f"Starting inventory extract (incremental={incremental})")

    # ── Paginate ──────────────────────────────
    while next_url:
        data = fetch_page(next_url, params if page_num == 1 else {})
        if data is None:
            log.error("Aborting — fetch failed.")
            break

        results = data.get("results", [])
        if not results:
            log.info("Empty results page — extract complete.")
            break

        all_raw.extend(results)
        parsed = [parse_record(r) for r in results]
        all_records.extend(parsed)

        pagination = data.get("pagination", {})
        next_url   = pagination.get("next_page_url")  # None on last page
        total      = pagination.get("total_records", "?")

        log.info(
            f"  Page {page_num} — {len(results)} records "
            f"({len(all_records)}/{total} total)"
        )

        page_num += 1
        time.sleep(0.25)   # ~240 req/min safe rate

    log.info(f"Extract complete — {len(all_records)} records across {page_num-1} pages")

    # ── Save raw JSON ─────────────────────────
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = OUTPUT_DIR / f"inventory_raw_{ts}.json"
    with open(raw_path, "w") as f:
        json.dump(all_raw, f, indent=2)
    log.info(f"Raw JSON saved → {raw_path}")

    # ── Build DataFrame ───────────────────────
    df = pd.DataFrame(all_records)

    if df.empty:
        log.warning("No records extracted — returning empty DataFrame")
        return df

    # Cast types
    df["expiry_date"]  = pd.to_datetime(df["expiry_date"], errors="coerce")
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")

    # ── Save processed CSV ────────────────────
    csv_path = PROCESSED_DIR / f"inventory_clean_{ts}.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Processed CSV saved → {csv_path}")

    # ── Summary ───────────────────────────────
    stockout_count  = df["stockout_risk_computed"].sum()
    mismatch_count  = df["kpi_mismatch"].sum()
    cold_chain_count = df["cold_chain_required"].sum()

    log.info("\n" + "═" * 50)
    log.info("  INVENTORY EXTRACT SUMMARY")
    log.info("═" * 50)
    log.info(f"  Total records      : {len(df):,}")
    log.info(f"  Stockout risk      : {stockout_count} products (days < {STOCKOUT_DAYS})")
    log.info(f"  Cold chain required: {cold_chain_count} products (≤8°C)")
    log.info(f"  KPI mismatches     : {mismatch_count} (API vs computed)")
    log.info(f"  Null product_ids   : {df['product_id'].isna().sum()}")
    log.info(f"  Date range         : {df['last_updated'].min()} → {df['last_updated'].max()}")
    log.info("═" * 50)

    # ── Persist run timestamp ─────────────────
    save_run_timestamp()

    return df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pharma Inventory API Extractor")
    parser.add_argument(
        "--full", action="store_true",
        help="Force full extract (ignore last run timestamp)"
    )
    args = parser.parse_args()

    df = extract_inventory(incremental=not args.full)

    if not df.empty:
        print("\n── Sample output (first 5 rows) ──")
        print(df[[
            "product_id", "brand_name", "quantity_on_hand",
            "days_of_stock_computed", "stockout_risk_computed", "cold_chain_required"
        ]].head().to_string(index=False))

        print("\n── Stockout risk products ──")
        at_risk = df[df["stockout_risk_computed"]][
            ["product_id", "brand_name", "quantity_on_hand", "days_of_stock_computed"]
        ]
        print(at_risk.to_string(index=False) if not at_risk.empty else "None — all clear ✓")
