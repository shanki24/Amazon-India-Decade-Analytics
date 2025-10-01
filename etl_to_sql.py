# scripts/etl_to_sql.py
import os
import pandas as pd
from urllib.parse import quote_plus
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
from pathlib import Path

load_dotenv()

DB_USER = os.getenv('DB_USER', 'root')
DB_PASS = quote_plus(os.getenv('DB_PASSWORD', ''))
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '3306')
DB_NAME = os.getenv('DB_NAME', 'amazon_analytics')

# Path to your schema file that creates tables if you want to reuse it
SCHEMA = "/Users/shashankshandilya/Desktop/amazon_decade_project/sql/schema.sql"

# ---------- Helpers ----------
def get_engine(connection_url):
    # pool_pre_ping helps with flaky DB connections
    return create_engine(connection_url, pool_pre_ping=True)

def run_sql_statements_from_file(conn, sql_path):
    if not Path(sql_path).exists():
        print(f"⚠️ schema file not found at {sql_path} -- skipping schema apply.")
        return
    sql = open(sql_path, "r", encoding="utf8").read()
    # naive split by ';' is ok if schema.sql is well-formed
    for s in [s.strip() for s in sql.split(";") if s.strip()]:
        conn.execute(text(s))

def get_column_types(engine, schema, table):
    q = text("""SELECT column_name, data_type FROM information_schema.columns
                WHERE table_schema=:schema AND table_name=:table""")
    with engine.begin() as c:
        rows = c.execute(q, {"schema": schema, "table": table}).fetchall()
    return {r[0]: r[1].lower() for r in rows}

def normalize_tinyint_strings(df, engine, schema, table):
    # Map common textual booleans to 1/0 for tinyint columns in DB
    try:
        col_types = get_column_types(engine, schema, table)
    except Exception:
        return
    mapping = {"true": 1, "false": 0, "TRUE": 1, "FALSE": 0, "True": 1, "False": 0,
               "1": 1, "0": 0, "NULL": None, "null": None, "": None}
    for col, dtype in col_types.items():
        if dtype == "tinyint" and col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map(lambda v: mapping.get(str(v).strip(), v))
                try:
                    # convert to nullable integer if possible
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                except Exception:
                    pass

def clean_decimal_columns(df, numeric_cols):
    """Remove thousands separators and non-numeric characters then coerce to numeric."""
    for col in numeric_cols:
        if col in df.columns:
            # Convert to string, strip anything not digit/dot/minus, then numeric
            df[col] = df[col].astype(str).str.replace(r'[^0-9.\-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

# ---------- Config / paths ----------
P = Path("/Users/shashankshandilya/Desktop/amazon_decade_project/cleaned")
prod_csv = P / "products_cleaned.csv"
txn_csv = P / "transactions_cleaned.csv"

CONN_URL_MASTER = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}"
CONN_URL_DB = f"{CONN_URL_MASTER}/{DB_NAME}"

# ---------- Ensure DB + schema ----------
eng_master = get_engine(CONN_URL_MASTER)
with eng_master.begin() as conn:
    conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}`"))
    conn.execute(text(f"USE `{DB_NAME}`"))
    # attempt to apply schema if present
    run_sql_statements_from_file(conn, SCHEMA)
print("DB/schema ensured (if schema.sql present it was applied)")

# ---------- Connect to DB ----------
engine = get_engine(CONN_URL_DB)
inspector = inspect(engine)

# ---------- Validate input CSV files ----------
if not prod_csv.exists():
    raise FileNotFoundError(f"Products CSV not found: {prod_csv}")
if not txn_csv.exists():
    raise FileNotFoundError(f"Transactions CSV not found: {txn_csv}")

# ---------- Load product catalog (small) ----------
print(f"Loading products from: {prod_csv}")
products_df = pd.read_csv(prod_csv, dtype=str, low_memory=False)
print(f"   -> products rows: {len(products_df)}")

# ---------- Load transactions in chunks (safe) ----------
print(f"Loading transactions from: {txn_csv} (chunked)")
chunksize = 200_000
reader = pd.read_csv(txn_csv, dtype=str, chunksize=chunksize, low_memory=False)
chunks = []
for i, ch in enumerate(reader, start=1):
    print(f"   reading chunk {i}, rows={len(ch)}")
    chunks.append(ch)
transactions_df = pd.concat(chunks, ignore_index=True)
print("   -> total transactions rows (raw):", len(transactions_df))

# ---------- Parse order_date if present ----------
if "order_date" in transactions_df.columns:
    transactions_df["order_date"] = pd.to_datetime(transactions_df["order_date"], errors="coerce")

# ---------- Build customers DF from transactions BEFORE any DB inserts ----------
print("Building customers dataframe from transactions (pre-upload) ...")
cust_candidate_cols = [
    "customer_id", "customer_city", "customer_state",
    "customer_tier", "customer_spending_tier", "customer_age_group", "is_prime_member"
]
cust_cols_existing = [c for c in cust_candidate_cols if c in transactions_df.columns]

if cust_cols_existing:
    cust_df = transactions_df[cust_cols_existing].drop_duplicates(subset=["customer_id"]).copy()
else:
    if "customer_id" not in transactions_df.columns:
        raise KeyError("No customer_id in transactions.csv; cannot build customers table.")
    cust_df = pd.DataFrame({"customer_id": transactions_df["customer_id"].astype(str).unique()})

# compute segmentation metrics (try final_amount variants)
amt_candidates = [c for c in ["final_amount_inr", "final_amount", "final_amount_inr_clean"] if c in transactions_df.columns]
amt_col = amt_candidates[0] if amt_candidates else None

if amt_col:
    transactions_df["_amt_num"] = pd.to_numeric(transactions_df[amt_col].astype(str).str.replace(r'[^0-9.\-]', '', regex=True), errors="coerce")
    total_orders_col = "transaction_id" if "transaction_id" in transactions_df.columns else "order_date"
    cust_metrics = transactions_df.groupby("customer_id").agg(
        lifetime_value=("_amt_num", "sum"),
        total_orders=(total_orders_col, "count"),
        avg_order_value=("_amt_num", "mean")
    ).reset_index()
    cust_df = cust_df.merge(cust_metrics, on="customer_id", how="left")
    transactions_df.drop(columns=["_amt_num"], inplace=True, errors=True)
else:
    total_orders_col = "transaction_id" if "transaction_id" in transactions_df.columns else "order_date"
    cust_metrics = transactions_df.groupby("customer_id").agg(
        lifetime_value=(total_orders_col, "count"),
        total_orders=(total_orders_col, "count"),
        avg_order_value=(total_orders_col, "count")
    ).reset_index().rename(columns={total_orders_col: "total_orders"})
    cust_df = cust_df.merge(cust_metrics, on="customer_id", how="left")

# Ensure customer_id is string (match DB VARCHAR)
cust_df["customer_id"] = cust_df["customer_id"].astype(str)

# ---------- Align DataFrame columns to DB tables (keep only DB columns) ----------
prod_cols_db = [c["name"] for c in inspector.get_columns("products")] if "products" in inspector.get_table_names() else []
txn_cols_db = [c["name"] for c in inspector.get_columns("transactions")] if "transactions" in inspector.get_table_names() else []
cust_cols_db = [c["name"] for c in inspector.get_columns("customers")] if "customers" in inspector.get_table_names() else []
time_cols_db = [c["name"] for c in inspector.get_columns("time_dimension")] if "time_dimension" in inspector.get_table_names() else []

# Keep only db-known columns (if schema wasn't created, this will keep all original columns)
if prod_cols_db:
    products_df = products_df[[c for c in products_df.columns if c in prod_cols_db]]
if txn_cols_db:
    transactions_df = transactions_df[[c for c in transactions_df.columns if c in txn_cols_db]]
if cust_cols_db:
    # ensure cust_df has all DB customer columns (add missing with NA)
    for c in cust_cols_db:
        if c not in cust_df.columns:
            cust_df[c] = pd.NA
    cust_df = cust_df[cust_cols_db]

# ---------- Normalize boolean-like tinyint columns based on DB schema ----------
normalize_tinyint_strings(products_df, engine, DB_NAME, "products")
normalize_tinyint_strings(transactions_df, engine, DB_NAME, "transactions")
normalize_tinyint_strings(cust_df, engine, DB_NAME, "customers")

# ---------- Clean decimal / numeric columns to avoid MySQL DECIMAL errors ----------
# Transaction numeric candidates (extend as needed)
txn_numeric_cols = [
    "original_price_inr", "discount_percent", "discounted_price_inr",
    "quantity", "subtotal_inr", "delivery_charges", "final_amount_inr",
    "product_weight_kg", "product_rating", "customer_rating",
    "final_amount", "original_price_inr_clean"
]
# Product numeric candidates
prod_numeric_cols = [
    "base_price_2015", "weight_kg", "rating", "price"
]

clean_decimal_columns = globals().get("clean_decimal_columns")
if not clean_decimal_columns:
    # define fallback if earlier cell not present (shouldn't happen)
    def clean_decimal_columns(df, numeric_cols):
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(r'[^0-9.\-]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')

clean_decimal_columns(transactions_df, txn_numeric_cols)
clean_decimal_columns(products_df, prod_numeric_cols)
# also ensure integer-like columns are proper dtype (if present)
for int_col in ["order_month", "order_quarter", "order_year", "delivery_days", "quantity"]:
    if int_col in transactions_df.columns:
        transactions_df[int_col] = pd.to_numeric(transactions_df[int_col], errors='coerce').astype("Int64")

# ---------- Safe TRUNCATE (disable FK checks while truncating) ----------
with engine.begin() as conn:
    print("   Disabling FK checks and truncating tables (transactions, customers, products, time_dimension) ...")
    conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
    for t in ["transactions", "customers", "products", "time_dimension"]:
        if t in inspector.get_table_names():
            print(f"     TRUNCATE {t}")
            conn.execute(text(f"TRUNCATE TABLE `{t}`;"))
    conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
    print("   Truncate complete and FK checks re-enabled.")

# ---------- Insert in parent-first order to satisfy foreign keys ----------
print("Uploading products -> products table")
if len(products_df):
    products_df.to_sql("products", engine, if_exists="append", index=False, chunksize=5000)
else:
    print("   products_df is empty (no rows to upload)")

print("Uploading customers -> customers table (must exist before transactions)")
if len(cust_df):
    # ensure customer_id column exists in cust_df and not null for primary key
    if "customer_id" not in cust_df.columns:
        raise KeyError("customer_id missing from cust_df; cannot populate customers table.")
    cust_df.to_sql("customers", engine, if_exists="append", index=False, chunksize=5000)
else:
    print("   ⚠️ cust_df is empty (no rows to upload)")

print("Uploading transactions -> transactions table")
# Now insert transactions (customers & products are already uploaded)
if len(transactions_df):
    # convert order_date column to naive datetime if necessary
    if "order_date" in transactions_df.columns:
        transactions_df["order_date"] = pd.to_datetime(transactions_df["order_date"], errors="coerce")
    transactions_df.to_sql("transactions", engine, if_exists="append", index=False, chunksize=5000)
else:
    print("   ⚠️ transactions_df empty - nothing to insert")

# ---------- Build and upload time_dimension from transactions date range ----------
print("Building time_dimension from transactions.order_date ...")
if "order_date" in transactions_df.columns and transactions_df["order_date"].notna().any():
    min_date = pd.to_datetime(transactions_df["order_date"].min()).floor("D")
    max_date = pd.to_datetime(transactions_df["order_date"].max()).ceil("D")
    all_dates = pd.date_range(start=min_date, end=max_date, freq="D")
    time_df = pd.DataFrame({"date_id": all_dates})
    time_df["year"] = time_df["date_id"].dt.year
    time_df["quarter"] = time_df["date_id"].dt.quarter
    time_df["month"] = time_df["date_id"].dt.month
    time_df["month_name"] = time_df["date_id"].dt.strftime("%B")
    time_df["week_of_year"] = time_df["date_id"].dt.isocalendar().week.astype(int)
    time_df["day_of_month"] = time_df["date_id"].dt.day
    time_df["day_name"] = time_df["date_id"].dt.strftime("%A")
    time_df["is_weekend"] = time_df["date_id"].dt.dayofweek.isin([5,6]).astype(int)
    if len(time_df):
        time_df.to_sql("time_dimension", engine, if_exists="append", index=False, chunksize=5000)
        print("✅ Time dimension table populated")
    else:
        print("⚠️ time_df empty - skipped time_dimension upload")
else:
    print("⚠️ order_date missing or all null - skipped time_dimension creation")

print("ETL finished: products, customers, transactions, time_dimension loaded (where applicable).")
