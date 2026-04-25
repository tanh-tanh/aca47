"""
scripts/forecast.py
====================
Production forecast script for Datathon 2026.
Uses the V8 Hybrid model (Seasonal + Growth + Partial DOW) 
which consistently achieves ~1.0M CV MAE and <800k on target submissions.
"""

import sys
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from data_utils import load_all, get_daily_revenue  # noqa: E402
from evaluate import make_submission                # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Configuration (V8 Hybrid Best Params)
# ══════════════════════════════════════════════════════════════════════════════
GROWTH_REV     = 1.0572
GROWTH_COGS    = 1.0478
BASE_DAYS_REV  = 389
BASE_DAYS_COGS = 361
DOW_ALPHA      = 0.8778   
MIN_YEAR       = 2012     


# ══════════════════════════════════════════════════════════════════════════════
# Core Functions
# ══════════════════════════════════════════════════════════════════════════════

def make_seasonal(df_tr, min_year):
    """
    Build normalized (month, day) seasonal profile + DOW correction factor.
    """
    sub = df_tr[df_tr['year'] >= int(min_year)].copy()
    ameans = sub.groupby('year')[['Revenue', 'COGS']].transform('mean')
    
    sub['rn'] = sub['Revenue'] / ameans['Revenue']
    sub['cn'] = sub['COGS']    / ameans['COGS']
    seas = sub.groupby(['month', 'day'])[['rn', 'cn']].mean().reset_index()
    
    # DOW residual ratio
    sub_m = sub.merge(seas, on=['month', 'day'], suffixes=('', '_s'), how='left')
    sub_m['dow'] = sub_m['date'].dt.dayofweek
    sub_m['rr'] = (sub_m['rn'] / sub_m['rn_s'].replace(0, np.nan)).clip(0.3, 3.0)
    sub_m['cr'] = (sub_m['cn'] / sub_m['cn_s'].replace(0, np.nan)).clip(0.3, 3.0)
    
    dow_factor = sub_m.groupby(['month', 'dow'])[['rr', 'cr']].mean().reset_index()
    return seas, dow_factor


def make_forecast(df_tr, df_v, gr, gc, br_days, bc_days, seas, dow_factor, alpha_dow=1.0):
    """
    Seasonal + growth + partial DOW forecast.
    """
    dv = df_v.copy()
    if 'Date' in dv.columns:
        dv['date'] = pd.to_datetime(dv['Date'])
    else:
        dv['date'] = pd.to_datetime(dv['date'])
        
    dv['month'] = dv['date'].dt.month
    dv['day']   = dv['date'].dt.day
    dv['dow']   = dv['date'].dt.dayofweek
    dv['year']  = dv['date'].dt.year
    dv['years_ahead'] = dv['year'] - df_tr['year'].max()

    dv = dv.merge(seas, on=['month', 'day'], how='left')
    dv = dv.merge(dow_factor, on=['month', 'dow'], how='left')
    
    for col in ['rn', 'cn', 'rr', 'cr']:
        dv[col] = dv[col].fillna(1.0)

    rr_blend = 1.0 + alpha_dow * (dv['rr'].values - 1.0)
    cr_blend = 1.0 + alpha_dow * (dv['cr'].values - 1.0)

    base_r = df_tr.tail(int(br_days))['Revenue'].mean()
    base_c = df_tr.tail(int(bc_days))['COGS'].mean()

    pred_r = base_r * (gr ** dv['years_ahead'].values) * dv['rn'].values * rr_blend
    pred_c = base_c * (gc ** dv['years_ahead'].values) * dv['cn'].values * cr_blend
    return pred_r, pred_c


# ══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading data...")
    dfs = load_all()
    train = get_daily_revenue(dfs).copy()
    test_dates = pd.to_datetime(dfs['submission']['Date']).to_frame()

    train['year']  = train['date'].dt.year
    train['month'] = train['date'].dt.month
    train['day']   = train['date'].dt.day
    train['dow']   = train['date'].dt.dayofweek
    train = train.sort_values("date").reset_index(drop=True)

    print(f"Train range: {train['date'].min().date()} -> {train['date'].max().date()}")
    
    print("\nEvaluating 2-fold CV (2021, 2022):")
    total_cv_mae = 0
    for val_yr in [2021, 2022]:
        df_tr = train[train['year'] < val_yr].copy()
        df_v  = train[train['year'] == val_yr].copy()
        val_dates = df_v[['date']].rename(columns={'date': 'Date'})
        
        seas_v, dow_v = make_seasonal(df_tr, MIN_YEAR)
        pr, pc = make_forecast(df_tr, val_dates, GROWTH_REV, GROWTH_COGS,
                               BASE_DAYS_REV, BASE_DAYS_COGS, seas_v, dow_v, DOW_ALPHA)
        
        mr = np.abs(df_v['Revenue'].values - pr).mean()
        mc = np.abs(df_v['COGS'].values    - pc).mean()
        total_mae = mr + mc
        total_cv_mae += total_mae
        print(f"  [{val_yr}] Rev={mr:,.0f} | COGS={mc:,.0f} | Total={total_mae:,.0f}")
        
    print(f"  => Average CV MAE: {total_cv_mae / 2:,.0f}")

    print("\nGenerating final submission...")
    seas_full, dow_full = make_seasonal(train, MIN_YEAR)
    pred_rev, pred_cogs = make_forecast(
        train, test_dates,
        GROWTH_REV, GROWTH_COGS,
        BASE_DAYS_REV, BASE_DAYS_COGS,
        seas_full, dow_full,
        DOW_ALPHA
    )

    # Constraints
    pred_rev  = np.clip(pred_rev, 0, None)
    pred_cogs = np.clip(pred_cogs, 0, None)
    mask = pred_rev < pred_cogs
    pred_rev[mask] = pred_cogs[mask] * 1.02

    sub_path = OUT_DIR / "submission.csv"
    sub = make_submission(test_dates['Date'], pred_rev, pred_cogs, str(sub_path))
    
    print(f"Saved to {sub_path}")
    print(f"  Rev Avg : {pred_rev.mean():,.0f}")
    print(f"  COGS Avg: {pred_cogs.mean():,.0f}")
    
    # Proxy evaluation
    try:
        sample = pd.read_csv(str(DATA_DIR / "sample_submission.csv"))
        sample["Date"] = pd.to_datetime(sample["Date"])
        sub2 = sub.copy()
        sub2["Date"] = pd.to_datetime(sub2["Date"])
        merged = sample.merge(sub2, on="Date", suffixes=("_s", "_p"))
        rv = float(np.abs(merged["Revenue_s"] - merged["Revenue"]).mean())
        cv = float(np.abs(merged["COGS_s"] - merged["COGS"]).mean())
        print(f"\nProxy MAE vs sample_submission: {rv+cv:,.0f}")
    except Exception as e:
        pass
        
    print("\nDone!")
