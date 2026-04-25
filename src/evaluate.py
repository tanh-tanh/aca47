"""
src/evaluate.py
Metrics đánh giá theo đúng yêu cầu cuộc thi: MAE, RMSE, R², MAPE.
"""

import numpy as np
import pandas as pd


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - actual)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def mape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1.0) -> float:
    """Mean Absolute Percentage Error. epsilon tránh chia cho 0."""
    return float(
        np.mean(np.abs(predicted - actual) / (np.abs(actual) + epsilon)) * 100
    )


def evaluate(actual: np.ndarray, predicted: np.ndarray, label: str = "") -> dict:
    """In và trả về dict metrics."""
    m = {
        "MAE":  mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "R2":   r2(actual, predicted),
        "MAPE": mape(actual, predicted),
    }
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}MAE  = {m['MAE']:,.0f}")
    print(f"{prefix}RMSE = {m['RMSE']:,.0f}")
    print(f"{prefix}R²   = {m['R2']:.4f}")
    print(f"{prefix}MAPE = {m['MAPE']:.2f}%")
    return m


def make_submission(dates: pd.Series,
                    revenue_pred: np.ndarray,
                    cogs_pred: np.ndarray,
                    path: str = "../outputs/submission.csv") -> pd.DataFrame:
    """
    Tạo file submission đúng format.
    Giữ nguyên thứ tự dates (không sort lại).
    """
    sub = pd.DataFrame({
        "Date":    pd.to_datetime(dates).dt.strftime("%Y-%m-%d"),
        "Revenue": np.round(revenue_pred, 2),
        "COGS":    np.round(cogs_pred, 2),
    })
    sub.to_csv(path, index=False)
    print(f"Submission saved -> {path}  ({len(sub)} rows)")
    return sub
