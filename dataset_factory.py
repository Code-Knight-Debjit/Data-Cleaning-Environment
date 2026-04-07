"""
dataset_factory.py
------------------
Generates (dirty_df, clean_df, metadata) triples for all 3 tasks.

Key design decisions:
  - Fixed random seeds per task → reproducible grader scores
  - clean_df is ALWAYS generated first, then dirt is injected
  - metadata carries ground-truth info the grader needs (e.g. which
    rows are real outliers vs valid extremes in Task 2)
  - No external files needed — everything is generated in memory
"""

from __future__ import annotations

import copy
import random
import string
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ── Reproducible seeds ────────────────────────────────────────────────────────

SEEDS = {
    "easy":   42,
    "medium": 137,
    "hard":   999,
}

# ── Return type ───────────────────────────────────────────────────────────────

@dataclass
class TaskDataset:
    """Everything the environment and grader need for one episode."""
    task_id: str
    dirty_df: pd.DataFrame
    clean_df: pd.DataFrame
    schema_hint: str                        # plain-English schema description
    total_dirty_cells: int                  # how many cells differ at episode start
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata keys used by graders:
    #   "outlier_rows"    (Task 2) — list of row indices that ARE true outliers
    #   "valid_extreme_rows" (Task 2) — valid rows that look extreme but must stay
    #   "canonical_columns"  (Task 3) — {alias: canonical_name} mapping
    #   "duplicate_row_ids"  (Task 3) — list of (original_idx, duplicate_idx) pairs


# ── Public API ────────────────────────────────────────────────────────────────

def make_dataset(task_id: str) -> TaskDataset:
    """Entry point. Call this from the environment's reset()."""
    if task_id == "easy":
        return _make_easy()
    elif task_id == "medium":
        return _make_medium()
    elif task_id == "hard":
        return _make_hard()
    else:
        raise ValueError(f"Unknown task_id: {task_id!r}. Must be easy/medium/hard.")


def count_dirty_cells(dirty_df: pd.DataFrame, clean_df: pd.DataFrame) -> int:
    """Number of cells that differ between dirty and clean DataFrames."""
    # Align on same dtypes for comparison
    d = dirty_df.astype(str).reset_index(drop=True)
    c = clean_df.astype(str).reset_index(drop=True)
    return int((d != c).sum().sum())


# ── Task 1: easy ─────────────────────────────────────────────────────────────
#
# 50-row sales CSV.
# Clean schema:
#   order_id (int), customer (str), product (str), category (str),
#   price (float, 2dp), quantity (int), order_date (YYYY-MM-DD),
#   region (str)
#
# Injected issues (29 dirty cells total):
#   • 10 wrong-type cells  — numeric column contains a word
#   • 8  missing values    — NaN in various columns
#   • 5  bad dates         — future year (2099-xx-xx)
#   • 6  whitespace cells  — leading/trailing spaces in string columns

def _make_easy() -> TaskDataset:
    rng = random.Random(SEEDS["easy"])
    np_rng = np.random.default_rng(SEEDS["easy"])

    n = 50
    categories = ["Electronics", "Clothing", "Home", "Sports", "Books"]
    regions    = ["North", "South", "East", "West"]
    products   = ["Widget A", "Widget B", "Gadget X", "Gadget Y", "Item Z"]
    customers  = [f"Customer_{i:03d}" for i in range(1, 31)]

    # ── Build clean DataFrame ────────────────────────────────────────────────
    clean = pd.DataFrame({
        "order_id":   range(1001, 1001 + n),
        "customer":   [rng.choice(customers) for _ in range(n)],
        "product":    [rng.choice(products)  for _ in range(n)],
        "category":   [rng.choice(categories) for _ in range(n)],
        "price":      np_rng.uniform(5.0, 500.0, n).round(2),
        "quantity":   np_rng.integers(1, 20, n),
        "order_date": _random_dates(np_rng, n, "2023-01-01", "2024-06-30"),
        "region":     [rng.choice(regions) for _ in range(n)],
    })
    clean["price"]    = clean["price"].astype(float)
    clean["quantity"] = clean["quantity"].astype(int)

    # ── Inject dirt ──────────────────────────────────────────────────────────
    dirty = clean.copy(deep=True).astype(object)

    injected: set[tuple[int, str]] = set()

    def pick_fresh(col: str, exclude: set) -> int:
        rows = [r for r in range(n) if (r, col) not in exclude]
        return rng.choice(rows)

    # 10 wrong-type cells in numeric columns
    bad_words = ["N/A", "unknown", "missing", "null", "TBD", "??", "-", "n/a", "none", "—"]
    for word, col in zip(bad_words, rng.choices(["price", "quantity"], k=10)):
        row = pick_fresh(col, injected)
        dirty.at[row, col] = word
        injected.add((row, col))

    # 8 missing values in various columns
    missing_cols = rng.choices(["customer", "product", "price", "quantity", "region"], k=8)
    for col in missing_cols:
        row = pick_fresh(col, injected)
        dirty.at[row, col] = np.nan
        injected.add((row, col))

    # 5 bad dates — far-future year
    bad_date_templates = [
        "2099-01-15", "2099-07-04", "2099-12-31", "2099-03-22", "2099-11-11"
    ]
    for bad_date in bad_date_templates:
        row = pick_fresh("order_date", injected)
        dirty.at[row, "order_date"] = bad_date
        injected.add((row, "order_date"))

    # 6 whitespace cells in string columns
    ws_cols = rng.choices(["customer", "product", "category", "region"], k=6)
    for col in ws_cols:
        row = pick_fresh(col, injected)
        orig = str(dirty.at[row, col])
        dirty.at[row, col] = f"  {orig}  "
        injected.add((row, col))

    dirty_cell_count = count_dirty_cells(dirty.astype(str), clean.astype(str))

    schema_hint = (
        "Sales orders dataset. Expected columns: "
        "order_id (integer), customer (string, no leading/trailing spaces), "
        "product (string, no spaces), category (one of: Electronics/Clothing/Home/Sports/Books), "
        "price (float, 2 decimal places, no text), "
        "quantity (integer, no text), "
        "order_date (YYYY-MM-DD format, year must be 2023 or 2024), "
        "region (one of: North/South/East/West, no spaces). "
        "No missing values allowed."
    )

    return TaskDataset(
        task_id="easy",
        dirty_df=dirty,
        clean_df=clean.astype(object),
        schema_hint=schema_hint,
        total_dirty_cells=dirty_cell_count,
        metadata={"injected_cells": list(injected)},
    )


# ── Task 2: medium ────────────────────────────────────────────────────────────
#
# 200-row customer transaction CSV.
# Clean schema:
#   tx_id (int), customer_id (int), amount (float), tx_date (YYYY-MM-DD),
#   category (str), country (str), status (str)
#
# Injected issues:
#   • 15 statistical outliers  — amount Z-score > 4.0  (should be removed/capped)
#   • 5  valid extremes        — genuinely large transactions, must NOT be removed
#   • 12 category typos        — slight misspellings

def _make_medium() -> TaskDataset:
    rng = random.Random(SEEDS["medium"])
    np_rng = np.random.default_rng(SEEDS["medium"])

    n = 200
    categories = ["Food", "Electronics", "Travel", "Healthcare", "Entertainment"]
    countries  = ["US", "UK", "CA", "AU", "DE"]
    statuses   = ["completed", "pending", "refunded"]

    # ── Build clean base ────────────────────────────────────────────────────
    # Normal transaction amounts: mean $150, sd $60, clipped to [5, 800]
    amounts = np_rng.normal(150, 60, n).clip(5, 800).round(2)

    clean = pd.DataFrame({
        "tx_id":       range(9001, 9001 + n),
        "customer_id": np_rng.integers(1, 501, n),
        "amount":      amounts,
        "tx_date":     _random_dates(np_rng, n, "2023-01-01", "2024-06-30"),
        "category":    [rng.choice(categories) for _ in range(n)],
        "country":     [rng.choice(countries)  for _ in range(n)],
        "status":      [rng.choice(statuses)   for _ in range(n)],
    })

    # ── Choose outlier rows (15) — will be injected with extreme amounts ─────
    all_rows = list(range(n))
    outlier_rows: list[int] = rng.sample(all_rows, 15)
    remaining    = [r for r in all_rows if r not in outlier_rows]

    # ── Choose valid extreme rows (5) — large but legitimate ─────────────────
    # These are NOT in outlier_rows; amounts are large (Z > 3) but real
    valid_extreme_rows: list[int] = rng.sample(remaining, 5)

    # ── Build dirty DataFrame ────────────────────────────────────────────────
    dirty = clean.copy(deep=True).astype(object)

    # Inject true outliers: very high or very low (Z > 4)
    for row in outlier_rows:
        if rng.random() > 0.3:
            dirty.at[row, "amount"] = round(rng.uniform(5000, 15000), 2)   # extreme high
        else:
            dirty.at[row, "amount"] = round(rng.uniform(-500, -10),  2)    # negative (impossible)

    # Inject valid extremes (in clean AND dirty — they stay)
    for row in valid_extreme_rows:
        valid_large = round(rng.uniform(900, 2000), 2)
        clean.at[row, "amount"] = valid_large
        dirty.at[row, "amount"] = valid_large

    # Inject 12 category typos
    typo_map: dict[str, str] = {
        "Electronics":   ["Electrnics", "Electronis", "Electonics"],
        "Food":          ["Foood", "Fod", "Fo0d"],
        "Travel":        ["Travle", "Trevel", "Travell"],
        "Healthcare":    ["Helthcare", "Healtcare", "Heathcare"],
        "Entertainment": ["Entertainmnt", "Entertainmet", "Entertainmen"],
    }
    injected_typo_rows: set[int] = set()
    typo_count = 0
    typo_cells: list[tuple[int, str, str]] = []   # (row, dirty_val, clean_val)

    for row in rng.sample(remaining, min(12, len(remaining))):
        if typo_count >= 12:
            break
        if row in injected_typo_rows:
            continue
        orig_cat = str(clean.at[row, "category"])
        misspellings = typo_map.get(orig_cat)
        if misspellings:
            bad = rng.choice(misspellings)
            dirty.at[row, "category"] = bad
            typo_cells.append((row, bad, orig_cat))
            injected_typo_rows.add(row)
            typo_count += 1

    dirty_cell_count = count_dirty_cells(dirty.astype(str), clean.astype(str))

    schema_hint = (
        "Customer transactions dataset. Expected columns: "
        "tx_id (integer), customer_id (integer 1–500), "
        "amount (float, must be positive; realistic range is $5–$2000; "
        "amounts above $2000 or below $0 are data errors), "
        "tx_date (YYYY-MM-DD), "
        "category (one of: Food/Electronics/Travel/Healthcare/Entertainment — exact spelling), "
        "country (two-letter code: US/UK/CA/AU/DE), "
        "status (one of: completed/pending/refunded). "
        "Note: some large transactions ($900–$2000) are legitimate — do not remove them. "
        "Only remove rows where the amount is clearly erroneous (negative or > $2000)."
    )

    return TaskDataset(
        task_id="medium",
        dirty_df=dirty,
        clean_df=clean.astype(object),
        schema_hint=schema_hint,
        total_dirty_cells=dirty_cell_count,
        metadata={
            "outlier_rows":      outlier_rows,
            "valid_extreme_rows": valid_extreme_rows,
            "typo_cells":        typo_cells,         # [(row, dirty_val, clean_val)]
        },
    )


# ── Task 3: hard ──────────────────────────────────────────────────────────────
#
# 400-row CSV merged from 3 fictional data sources.
# Each source uses different column names for the same concepts.
# Issues:
#   • Inconsistent column naming (3 aliases per concept)
#   • Mixed date formats across sources (ISO, US, EU)
#   • 30 duplicate rows (exact and near-duplicate)
#   • No schema documentation — agent must infer canonical form
#
# Canonical schema (what the agent must produce):
#   record_id, customer_id, full_name, email, amount,
#   currency, purchase_date (YYYY-MM-DD), product_name, region

_CANONICAL_COLS = [
    "record_id", "customer_id", "full_name", "email",
    "amount", "currency", "purchase_date", "product_name", "region",
]

# Column aliases per source
_SOURCE_ALIASES = {
    "source_a": {
        "record_id":    "record_id",
        "customer_id":  "cust_id",
        "full_name":    "name",
        "email":        "email_address",
        "amount":       "sale_amount",
        "currency":     "ccy",
        "purchase_date":"date",
        "product_name": "item",
        "region":       "territory",
    },
    "source_b": {
        "record_id":    "id",
        "customer_id":  "customer_id",
        "full_name":    "full_name",
        "email":        "contact_email",
        "amount":       "value",
        "currency":     "currency",
        "purchase_date":"purchase_date",
        "product_name": "product",
        "region":       "area",
    },
    "source_c": {
        "record_id":    "RecordID",
        "customer_id":  "CustomerID",
        "full_name":    "CustomerName",
        "email":        "Email",
        "amount":       "Amount",
        "currency":     "Currency",
        "purchase_date":"PurchaseDate",
        "product_name": "ProductName",
        "region":       "Region",
    },
}

# Date format used by each source
_SOURCE_DATE_FORMATS = {
    "source_a": "%Y-%m-%d",   # ISO: 2023-04-15
    "source_b": "%m/%d/%Y",   # US:  04/15/2023
    "source_c": "%d.%m.%Y",   # EU:  15.04.2023
}

def _make_hard() -> TaskDataset:
    rng = random.Random(SEEDS["hard"])
    np_rng = np.random.default_rng(SEEDS["hard"])

    currencies = ["USD", "EUR", "GBP"]
    regions    = ["APAC", "EMEA", "AMER", "LATAM"]
    products   = [
        "Pro Subscription", "Enterprise License", "Support Package",
        "Training Course", "Hardware Bundle", "Consulting Day",
    ]

    # Helper: generate a block of rows for one source
    def _source_block(source: str, n: int, id_start: int) -> pd.DataFrame:
        aliases   = _SOURCE_ALIASES[source]
        date_fmt  = _SOURCE_DATE_FORMATS[source]
        cust_ids  = np_rng.integers(2001, 3001, n)
        amounts   = np_rng.uniform(100, 5000, n).round(2)
        iso_dates = _random_dates(np_rng, n, "2022-01-01", "2024-06-30")

        # Format dates in source-specific format
        formatted_dates = [
            pd.to_datetime(d).strftime(date_fmt)
            for d in iso_dates
        ]

        names  = [_random_name(rng)  for _ in range(n)]
        emails = [_name_to_email(nm) for nm in names]

        data = {
            aliases["record_id"]:    range(id_start, id_start + n),
            aliases["customer_id"]:  cust_ids.tolist(),
            aliases["full_name"]:    names,
            aliases["email"]:        emails,
            aliases["amount"]:       amounts.tolist(),
            aliases["currency"]:     [rng.choice(currencies) for _ in range(n)],
            aliases["purchase_date"]: formatted_dates,
            aliases["product_name"]: [rng.choice(products)  for _ in range(n)],
            aliases["region"]:       [rng.choice(regions)   for _ in range(n)],
        }
        return pd.DataFrame(data)

    # Three sources, ~133 rows each (total ~400)
    block_a = _source_block("source_a", 134, id_start=1)
    block_b = _source_block("source_b", 133, id_start=135)
    block_c = _source_block("source_c", 133, id_start=268)

    # ── Canonical (clean) dataframe ─────────────────────────────────────────
    def _to_canonical(df: pd.DataFrame, source: str) -> pd.DataFrame:
        rev = {v: k for k, v in _SOURCE_ALIASES[source].items()}
        renamed = df.rename(columns=rev)
        # Normalise date to YYYY-MM-DD
        renamed["purchase_date"] = pd.to_datetime(
            renamed["purchase_date"],
            format=_SOURCE_DATE_FORMATS[source],
        ).dt.strftime("%Y-%m-%d")
        return renamed[_CANONICAL_COLS]

    clean_a = _to_canonical(block_a, "source_a")
    clean_b = _to_canonical(block_b, "source_b")
    clean_c = _to_canonical(block_c, "source_c")
    clean   = pd.concat([clean_a, clean_b, clean_c], ignore_index=True)
    clean["record_id"] = range(1, len(clean) + 1)

    # ── Dirty dataframe = concat of raw source blocks ────────────────────────
    # (columns are still in aliased form, dates in source-specific format)
    dirty = pd.concat([block_a, block_b, block_c], ignore_index=True)

    # ── Inject 30 duplicate rows ─────────────────────────────────────────────
    n_clean = len(dirty)
    sampled_orig = rng.sample(range(n_clean), 30)
    duplicate_rows_to_inject: list[pd.DataFrame] = []
    duplicate_pairs: list[tuple[int, int]] = []

    for orig_idx in sampled_orig:
        dup = dirty.iloc[[orig_idx]].copy()
        # Near-duplicate: 40% chance of a minor field change
        if rng.random() < 0.4:
            # Slightly alter the amount (±1%)
            col_amount = list(_SOURCE_ALIASES["source_a"].values())[4]  # 'sale_amount'
            # Find which column name is 'amount-like' in this row's source
            # Since we concat all sources, each row might have NaN in other sources' cols.
            # Simpler: just modify the raw value in the only non-null amount column.
            for amt_col in ["sale_amount", "value", "Amount"]:
                if amt_col in dup.columns and pd.notna(dup.iloc[0].get(amt_col)):
                    old_val = dup.at[dup.index[0], amt_col]
                    dup.at[dup.index[0], amt_col] = round(float(old_val) * rng.uniform(0.99, 1.01), 2)
                    break
        duplicate_rows_to_inject.append(dup)
        duplicate_pairs.append((orig_idx, n_clean + len(duplicate_pairs)))

    dirty = pd.concat([dirty] + duplicate_rows_to_inject, ignore_index=True)

    # Shuffle so duplicates aren't obviously at the bottom
    dirty = dirty.sample(frac=1, random_state=SEEDS["hard"]).reset_index(drop=True)

    # Build canonical alias lookup for grader
    canonical_lookup: dict[str, str] = {}
    for source, aliases in _SOURCE_ALIASES.items():
        for canonical, alias in aliases.items():
            canonical_lookup[alias] = canonical

    dirty_cell_count = len(dirty) * len(_CANONICAL_COLS)  # hard task: whole-df scope

    schema_hint = (
        "Merged dataset from 3 sources with inconsistent schemas. "
        "Your goal is to produce a single clean DataFrame with these canonical columns: "
        "record_id (integer, unique), customer_id (integer), full_name (string), "
        "email (string), amount (float), currency (one of: USD/EUR/GBP), "
        "purchase_date (YYYY-MM-DD), product_name (string), region (one of: APAC/EMEA/AMER/LATAM). "
        "Column names in the raw data vary by source (e.g. 'cust_id', 'customer_id', 'CustomerID' "
        "all mean customer_id). Date formats also vary (ISO, US MM/DD/YYYY, EU DD.MM.YYYY). "
        "There are also ~30 duplicate rows (some exact, some near-duplicate). "
        "Remove duplicates, normalise all column names and date formats."
    )

    return TaskDataset(
        task_id="hard",
        dirty_df=dirty,
        clean_df=clean.astype(object),
        schema_hint=schema_hint,
        total_dirty_cells=dirty_cell_count,
        metadata={
            "canonical_columns":  _CANONICAL_COLS,
            "canonical_lookup":   canonical_lookup,   # alias → canonical name
            "source_aliases":     _SOURCE_ALIASES,
            "source_date_formats": _SOURCE_DATE_FORMATS,
            "duplicate_pairs":    duplicate_pairs,    # (original_idx, dup_idx) in pre-shuffle dirty
            "n_clean_rows":       len(clean),
        },
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _random_dates(
    rng: np.random.Generator,
    n: int,
    start: str,
    end: str,
) -> list[str]:
    """Generate n random ISO-format date strings between start and end."""
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)
    delta_days = (end_ts - start_ts).days
    offsets = rng.integers(0, delta_days, n)
    return [
        (start_ts + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in offsets
    ]


_FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Karen", "Leo", "Mia", "Nathan", "Olivia", "Paul",
    "Quinn", "Rosa", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier",
    "Yuki", "Zara",
]

_LAST_NAMES = [
    "Smith", "Jones", "Williams", "Brown", "Taylor", "Davies", "Evans",
    "Wilson", "Thomas", "Roberts", "Johnson", "Lee", "Martin", "Garcia",
    "Martinez", "Anderson", "Thompson", "White", "Harris", "Clark",
]


def _random_name(rng: random.Random) -> str:
    return f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"


def _name_to_email(name: str) -> str:
    first, last = name.lower().split()
    domains = ["example.com", "mail.com", "inbox.net", "corp.io"]
    return f"{first}.{last}@{domains[hash(name) % len(domains)]}"


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for task_id in ("easy", "medium", "hard"):
        ds = make_dataset(task_id)
        print(f"\n{'─'*60}")
        print(f"Task: {task_id.upper()}")
        print(f"  dirty shape : {ds.dirty_df.shape}")
        print(f"  clean shape : {ds.clean_df.shape}")
        print(f"  dirty cells : {ds.total_dirty_cells}")
        print(f"  schema hint : {ds.schema_hint[:80]}…")
        print(f"  metadata keys: {list(ds.metadata.keys())}")
        if task_id == "easy":
            print(f"\n  Sample dirty rows (price/quantity col):")
            mask = ds.dirty_df["price"].astype(str).str.contains(
                r"[a-zA-Z]|nan", na=True
            )
            print(ds.dirty_df[mask][["order_id","price","quantity"]].head(3).to_string(index=False))
        if task_id == "medium":
            print(f"\n  Outlier rows (first 5): {ds.metadata['outlier_rows'][:5]}")
            print(f"  Valid extreme rows:     {ds.metadata['valid_extreme_rows']}")
        if task_id == "hard":
            print(f"\n  Raw column names: {list(ds.dirty_df.columns)}")
            print(f"  Duplicate pairs (first 3): {ds.metadata['duplicate_pairs'][:3]}")