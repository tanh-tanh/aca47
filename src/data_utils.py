"""
src/data_utils.py
Helpers để load và merge các bảng dữ liệu.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_all():
    """Load toàn bộ CSV, trả về dict {tên: DataFrame}."""
    files = {
        "sales":       ("sales.csv",          {"Date": "date"}),
        "orders":      ("orders.csv",          {"order_date": "date"}),
        "order_items": ("order_items.csv",     {}),
        "customers":   ("customers.csv",       {"signup_date": "date"}),
        "products":    ("products.csv",        {}),
        "promotions":  ("promotions.csv",      {"start_date": "date", "end_date": "date"}),
        "geography":   ("geography.csv",       {}),
        "payments":    ("payments.csv",        {}),
        "shipments":   ("shipments.csv",       {"ship_date": "date", "delivery_date": "date"}),
        "returns":     ("returns.csv",         {"return_date": "date"}),
        "reviews":     ("reviews.csv",         {"review_date": "date"}),
        "inventory":   ("inventory.csv",       {"snapshot_date": "date"}),
        "web_traffic": ("web_traffic.csv",     {"date": "date"}),
        "submission":  ("sample_submission.csv", {"Date": "date"}),
    }
    dfs = {}
    for name, (fname, date_cols) in files.items():
        path = DATA_DIR / fname
        if not path.exists():
            print(f"[warn] {fname} not found, skipping")
            continue
        df = pd.read_csv(path, low_memory=False)
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        dfs[name] = df
    return dfs


def get_daily_revenue(dfs):
    """Trả về sales DataFrame với index là Date, đã sort."""
    s = dfs["sales"].copy()
    s = s.rename(columns={"Date": "date"})
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date").reset_index(drop=True)
    return s


def get_orders_with_geo(dfs):
    """Join orders + geography để có region/city."""
    orders = dfs["orders"].copy()
    geo    = dfs["geography"][["zip", "region", "district"]].drop_duplicates("zip")
    return orders.merge(geo, on="zip", how="left")


def get_order_items_with_products(dfs):
    """Join order_items + products để có category, segment, size."""
    oi = dfs["order_items"].copy()
    pr = dfs["products"][["product_id", "category", "segment", "size", "price", "cogs"]]
    return oi.merge(pr, on="product_id", how="left")


def get_returns_with_products(dfs):
    """Join returns + products."""
    r  = dfs["returns"].copy()
    pr = dfs["products"][["product_id", "category", "segment", "size"]]
    return r.merge(pr, on="product_id", how="left")


def get_web_traffic_daily(dfs):
    """Aggregate web_traffic theo ngày (sum sessions, mean bounce_rate, v.v.)."""
    wt = dfs["web_traffic"].copy()
    daily = wt.groupby("date").agg(
        sessions=("sessions", "sum"),
        unique_visitors=("unique_visitors", "sum"),
        page_views=("page_views", "sum"),
        bounce_rate=("bounce_rate", "mean"),
        avg_session_duration_sec=("avg_session_duration_sec", "mean"),
    ).reset_index()
    return daily
