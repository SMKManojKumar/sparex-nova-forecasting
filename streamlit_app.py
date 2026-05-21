# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import io, re
from datetime import datetime, timedelta

# ── optional ML imports ──────────────────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    HAS_SK = True
except ImportError:
    HAS_SK = False

# ═════════════════════════════════════════════════════════════════════════════
#  Regex hints for column detection
# ═════════════════════════════════════════════════════════════════════════════
_DATE_HINTS  = re.compile(r"date|month|period|time|week|year|quarter", re.I)
_QTY_HINTS   = re.compile(r"demand|qty|quantity|sales|usage|consumption|amount|vol", re.I)
_PART_HINTS  = re.compile(r"part|item|sku|product|component|material|code", re.I)

def _detect_cols(df: pd.DataFrame):
    cols = list(df.columns)
    date_col = part_col = qty_col = None
    for c in cols:
        if _DATE_HINTS.search(c) and date_col is None:
            date_col = c
        elif _PART_HINTS.search(c) and part_col is None:
            part_col = c
        elif _QTY_HINTS.search(c) and qty_col is None:
            qty_col = c

    if date_col is None:
        date_col = cols[0]
    if qty_col is None:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        qty_col = num_cols[-1] if num_cols else cols[-1]
    if part_col is None:
        obj_cols = df.select_dtypes(include="object").columns.tolist()
        obj_cols = [c for c in obj_cols if c not in (date_col, qty_col)]
        part_col = obj_cols[0] if obj_cols else None

    return date_col, part_col, qty_col

def _forecast_part(values: list, steps: int):
    n = len(values)
    X = np.arange(n).reshape(-1, 1)
    y = np.array(values, dtype=float)

    warnings = []
    if not HAS_SK:
        slope = (y[-1] - y[0]) / max(n - 1, 1) if n > 1 else 0
        pred = y.copy()
        fut = [max(0, round(y[-1] + slope * (i + 1), 2)) for i in range(steps)]
        mae = round(float(np.mean(np.abs(y - pred))), 2)
        return pred.tolist(), fut, mae, "Linear (fallback)", warnings

    if n >= 8:
        model = RandomForestRegressor(n_estimators=60, random_state=42)
        model_name = "Random Forest"
    else:
        model = LinearRegression()
        model_name = "Linear Regression"
        if n < 4:
            warnings.append(f"Very few records ({n}) — forecast may be inaccurate.")

    model.fit(X, y)
    hist_pred = model.predict(X).tolist()
    fut_X = np.arange(n, n + steps).reshape(-1, 1)
    fut_vals = [max(0, round(v, 2)) for v in model.predict(fut_X).tolist()]
    mae = round(mean_absolute_error(y, hist_pred), 2)
    return hist_pred, fut_vals, mae, model_name, warnings

def _status_label(avg_hist: float, avg_fore: float) -> str:
    if avg_hist == 0:
        return "Demand Stable"
    ratio = avg_fore / avg_hist
    if ratio > 1.4:
        return "High Demand Expected"
    if ratio < 0.5:
        return "Overstock Risk"
    if ratio < 0.75:
        return "Risk of Stockout"
    if ratio > 1.15:
        return "Purchase Recommended"
    return "Demand Stable"

def run_forecast(df: pd.DataFrame, steps: int = 6) -> dict:
    warnings = []
    date_col, part_col, qty_col = _detect_cols(df)
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)

    if date_col and date_col in df.columns:
        try:
            df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
            df = df.sort_values(date_col)
        except Exception:
            pass

    parts_result = {}
    if part_col and part_col in df.columns:
        groups = df.groupby(part_col)
    else:
        warnings.append("No part/SKU column detected — treating entire dataset as one part.")
        df["__part__"] = "All Parts"
        part_col = "__part__"
        groups = df.groupby(part_col)

    for pname, grp in groups:
        pname = str(pname)
        vals = grp[qty_col].tolist()
        if len(vals) < 2:
            parts_result[pname] = {"error": "Not enough data (< 2 records)."}
            continue

        hist_dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in grp[date_col]]
        try:
            last_date = grp[date_col].iloc[-1]
            delta = (grp[date_col].iloc[-1] - grp[date_col].iloc[-2]) if len(grp) > 1 else timedelta(days=30)
            fut_dates = [str((last_date + delta * (i + 1)).date()) for i in range(steps)]
        except Exception:
            fut_dates = [f"F{i+1}" for i in range(steps)]

        hist_pred, fut_vals, mae, model_name, part_warns = _forecast_part(vals, steps)
        warnings.extend(part_warns)

        avg_hist = round(float(np.mean(vals)), 2)
        avg_fore = round(float(np.mean(fut_vals)), 2)
        status = _status_label(avg_hist, avg_fore)

        parts_result[pname] = {
            "hist_dates": hist_dates,
            "hist_actual": [round(v, 2) for v in vals],
            "hist_pred": [round(v, 2) for v in hist_pred],
            "fut_dates": fut_dates,
            "fut_vals": fut_vals,
            "avg_hist": avg_hist,
            "avg_fore": avg_fore,
            "mae": mae,
            "model": model_name,
            "status": status,
        }

    return {"parts": parts_result, "warnings": warnings}

# ═════════════════════════════════════════════════════════════════════════════
#  Streamlit UI
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="SPREX NOVA Forecast", layout="wide")

st.title("📊 SPREX NOVA — Demand Forecasting")
st.markdown("Upload your dataset (CSV/XLSX) and get forecasts with ML models.")

uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx"])
steps = st.slider("Forecast steps ahead", 1, 12, 6)

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"File `{uploaded_file.name}` uploaded successfully!")
        st.dataframe(df.head())

        with st.spinner("🔮 Running forecast..."):
            result = run_forecast(df, steps=steps)

        for pname, pdata in result["parts"].items():
            st.subheader(f"📦 Part: {pname}")
            if "error" in pdata:
                st.error(pdata["error"])
                continue

            st.write(f"**Model:** {pdata['model']} | **MAE:** {pdata['mae']} | **Status:** {pdata['status']}")

            # Historical vs Predicted
            st.line_chart(pd.DataFrame({
                "Actual": pdata["hist_actual"],
                "Predicted": pdata["hist_pred"]
            }, index=pdata["hist_dates"]))

            # Future forecast
            st.bar_chart(pd.DataFrame({
                "Forecast": pdata["fut_vals"]
            }, index=pdata["fut_dates"]))

        if result["warnings"]:
            st.warning("⚠️ " + " | ".join(result["warnings"]))

    except Exception as e:
        st.error(f"Forecast failed: {e}")
