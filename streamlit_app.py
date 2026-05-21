"""
SPREX NOVA — Streamlit App
Full conversion of Flask app.py → Streamlit
Preserves: ML engine, password hashing, in-memory DB, all UI pages, all logic.
Run: streamlit run streamlit_app.py
"""

import os, json, io, hashlib, secrets, re
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── optional ML imports (graceful fallback) ──────────────────────────────────
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    HAS_SK = True
except ImportError:
    HAS_SK = False

# ── optional openpyxl ────────────────────────────────────────────────────────
try:
    import openpyxl  # noqa: F401
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

# ═════════════════════════════════════════════════════════════════════════════
#  Page config  (must be FIRST streamlit call)
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SPREX NOVA — Forecasting The Future of Parts",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════════════════════
#  In-memory "database" (stored in st.session_state so it persists per session)
# ═════════════════════════════════════════════════════════════════════════════
if "_DB" not in st.session_state:
    st.session_state["_DB"] = {"users": {}, "uploads": {}, "notifs": {}}

_DB = st.session_state["_DB"]

# ─── Auth session state ───────────────────────────────────────────────────────
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "page" not in st.session_state:
    st.session_state["page"] = "login"
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"
if "_flash" not in st.session_state:
    st.session_state["_flash"] = []   # list of (category, message)

# ─── simple counter ids ───────────────────────────────────────────────────────
def _next_id(table: str) -> str:
    existing = [int(k) for k in _DB[table] if str(k).isdigit()]
    return str(max(existing, default=0) + 1)

# ─── password hashing ─────────────────────────────────────────────────────────
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# ─── flash helpers ────────────────────────────────────────────────────────────
def flash(message: str, category: str = "info"):
    st.session_state["_flash"].append((category, message))

def _show_flashes():
    flashes = st.session_state.get("_flash", [])
    if not flashes:
        return
    for cat, msg in flashes:
        if cat == "success":
            st.success(msg)
        elif cat == "error":
            st.error(msg)
        elif cat == "warning":
            st.warning(msg)
        else:
            st.info(msg)
    st.session_state["_flash"] = []

# ─── nav helper ───────────────────────────────────────────────────────────────
def nav(page: str):
    st.session_state["page"] = page
    st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
#  Auth helpers
# ═════════════════════════════════════════════════════════════════════════════
def get_current_user():
    uid = st.session_state.get("user_id")
    if not uid:
        return None
    return next((u for u in _DB["users"].values() if u["id"] == uid), None)

def unread_notif_count(user_id: str) -> int:
    return sum(
        1 for n in _DB["notifs"].values()
        if n["user_id"] == user_id and not n["read"]
    )

# ═════════════════════════════════════════════════════════════════════════════
#  Notification helper
# ═════════════════════════════════════════════════════════════════════════════
def _add_notif(user_id: str, ntype: str, message: str):
    nid = _next_id("notifs")
    _DB["notifs"][nid] = {
        "id": nid, "user_id": user_id, "ntype": ntype,
        "message": message, "read": False,
        "created_at": datetime.utcnow()
    }

# ═════════════════════════════════════════════════════════════════════════════
#  ML Forecast engine  (ORIGINAL — unchanged)
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

    # fallback: first col → date, last numeric → qty, first string → part
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
    """Forecast `steps` periods ahead for a single part's demand series."""
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
        grp = grp.sort_values(date_col) if date_col in grp.columns else grp
        vals = grp[qty_col].tolist()
        if len(vals) < 2:
            parts_result[pname] = {"error": "Not enough data (< 2 records)."}
            continue

        if date_col in grp.columns:
            try:
                hist_dates = [str(d.date()) for d in grp[date_col]]
            except Exception:
                hist_dates = [str(v) for v in grp[date_col]]
        else:
            hist_dates = [f"P{i+1}" for i in range(len(vals))]

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
            "hist_dates":  hist_dates,
            "hist_actual": [round(v, 2) for v in vals],
            "hist_pred":   [round(v, 2) for v in hist_pred],
            "fut_dates":   fut_dates,
            "fut_vals":    fut_vals,
            "avg_hist":    avg_hist,
            "avg_fore":    avg_fore,
            "mae":         mae,
            "model":       model_name,
            "status":      status,
        }

    return {"parts": parts_result, "warnings": warnings}

# ═════════════════════════════════════════════════════════════════════════════
#  CSS — Theme-aware styling (Light / Dark / Neon)
# ═════════════════════════════════════════════════════════════════════════════
THEME_CSS = {
    "light": {
        "--bg": "#f0f4ff", "--surf": "#ffffff", "--surf2": "#f5f7ff",
        "--accent": "#2563eb", "--asoft": "rgba(37,99,235,.08)",
        "--txt": "#0f172a", "--muted": "#64748b", "--bdr": "#e2e8f0",
        "--red": "#dc2626", "--green": "#16a34a", "--amber": "#d97706",
        "--sky": "#0284c7", "--sb-bg": "#1e293b", "--sb-txt": "#cbd5e1",
    },
    "dark": {
        "--bg": "#0d1117", "--surf": "#161b22", "--surf2": "#21262d",
        "--accent": "#60a5fa", "--asoft": "rgba(96,165,250,.10)",
        "--txt": "#e6edf3", "--muted": "#8b949e", "--bdr": "#30363d",
        "--red": "#f85149", "--green": "#3fb950", "--amber": "#d29922",
        "--sky": "#58a6ff", "--sb-bg": "#010409", "--sb-txt": "#8b949e",
    },
    "neon": {
        "--bg": "#05071a", "--surf": "#0b0f2a", "--surf2": "#101530",
        "--accent": "#38bdf8", "--asoft": "rgba(56,189,248,.08)",
        "--txt": "#e0f2fe", "--muted": "#4a6080", "--bdr": "#1a2540",
        "--red": "#f472b6", "--green": "#34d399", "--amber": "#fbbf24",
        "--sky": "#38bdf8", "--sb-bg": "#020510", "--sb-txt": "#4a6080",
    },
}

def _inject_css():
    t = st.session_state.get("theme", "light")
    v = THEME_CSS[t]
    acc = v["--accent"]
    bg  = v["--bg"]
    surf = v["--surf"]
    surf2 = v["--surf2"]
    txt  = v["--txt"]
    muted = v["--muted"]
    bdr  = v["--bdr"]
    sb_bg = v["--sb-bg"]
    sb_txt = v["--sb-txt"]
    red  = v["--red"]
    green = v["--green"]
    amber = v["--amber"]

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@700;800;900&display=swap');

/* ── Root & Body ── */
html, body, [data-testid="stAppViewContainer"] {{
    background: {bg} !important;
    color: {txt} !important;
    font-family: 'Inter', sans-serif !important;
}}
[data-testid="stAppViewContainer"] > .main {{
    background: {bg} !important;
}}
[data-testid="block-container"] {{
    padding: 1.5rem 2rem !important;
    max-width: 1200px;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {sb_bg} !important;
    border-right: 1px solid {bdr};
}}
[data-testid="stSidebar"] * {{
    color: {sb_txt} !important;
}}
[data-testid="stSidebar"] .stButton button {{
    background: transparent !important;
    color: {sb_txt} !important;
    border: none !important;
    text-align: left !important;
    width: 100% !important;
    padding: 10px 14px !important;
    border-radius: 8px !important;
    font-size: 0.88rem !important;
    transition: all .18s !important;
}}
[data-testid="stSidebar"] .stButton button:hover {{
    background: rgba(255,255,255,.07) !important;
    color: #fff !important;
}}

/* ── Cards ── */
.sn-card {{
    background: {surf};
    border: 1px solid {bdr};
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,.06);
}}

/* ── Stat cards ── */
.sn-stat-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 20px;
}}
.sn-stat {{
    background: {surf};
    border: 1px solid {bdr};
    border-radius: 14px;
    padding: 18px 20px;
    display: flex;
    align-items: center;
    gap: 14px;
}}
.sn-stat-ico {{
    width: 42px; height: 42px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    background: {v['--asoft']};
    color: {acc};
    flex-shrink: 0;
}}
.sn-stat-val {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.7rem; font-weight: 800;
    color: {txt}; line-height: 1;
}}
.sn-stat-lbl {{
    font-size: 12px; color: {muted}; margin-top: 3px;
}}

/* ── Hero / page header ── */
.sn-ph {{
    margin-bottom: 22px;
}}
.sn-ph h1 {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.6rem; font-weight: 800;
    color: {txt}; margin: 0 0 4px;
}}
.sn-ph p {{ font-size: 14px; color: {muted}; margin: 0; }}

/* ── About hero ── */
.about-hero {{
    background: linear-gradient(135deg, {acc}, #7c3aed);
    border-radius: 20px;
    padding: 42px 32px;
    text-align: center;
    color: #fff;
    margin-bottom: 20px;
}}
.about-hero h1 {{
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem; font-weight: 900; margin-bottom: 10px;
}}
.about-hero p {{ opacity: .9; font-size: 1rem; max-width: 520px; margin: 0 auto 20px; }}

/* ── Feature grid ── */
.feat-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 14px;
    margin-bottom: 20px;
}}
.feat {{
    background: {surf};
    border: 1px solid {bdr};
    border-radius: 14px;
    padding: 18px;
}}
.feat-ico {{ font-size: 1.5rem; margin-bottom: 8px; color: {acc}; font-weight: 900; }}
.feat-t {{ font-weight: 700; font-size: .9rem; margin-bottom: 5px; color: {txt}; }}
.feat-d {{ font-size: 12.5px; color: {muted}; line-height: 1.6; }}

/* ── Status chips ── */
.chip {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
}}
.chip-green  {{ background: rgba(22,163,74,.1); color: {green}; }}
.chip-red    {{ background: rgba(220,38,38,.1); color: {red}; }}
.chip-amber  {{ background: rgba(217,119,6,.1); color: {amber}; }}
.chip-blue   {{ background: {v['--asoft']}; color: {acc}; }}
.chip-sky    {{ background: rgba(2,132,199,.1); color: {v['--sky']}; }}
.chip-grey   {{ background: {surf2}; color: {muted}; }}
.chip-purple {{ background: rgba(124,58,237,.1); color: #7c3aed; }}

/* ── Buttons ── */
.stButton > button {{
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all .18s !important;
}}
.stButton > button[kind="primary"] {{
    background: {acc} !important;
    color: #fff !important;
    border: none !important;
}}
.stButton > button[kind="secondary"] {{
    background: {surf2} !important;
    color: {txt} !important;
    border: 1px solid {bdr} !important;
}}

/* ── Inputs ── */
.stTextInput input, .stSelectbox select, .stTextArea textarea {{
    background: {surf} !important;
    border: 1px solid {bdr} !important;
    color: {txt} !important;
    border-radius: 10px !important;
}}
.stTextInput input:focus, .stSelectbox select:focus {{
    border-color: {acc} !important;
    box-shadow: 0 0 0 3px {v['--asoft']} !important;
}}

/* ── Auth box ── */
.auth-wrap {{
    min-height: 90vh;
    display: flex;
    align-items: center;
    justify-content: center;
}}
.auth-box {{
    background: {surf};
    border: 1px solid {bdr};
    border-radius: 20px;
    padding: 36px 38px;
    width: 100%;
    max-width: 420px;
    box-shadow: 0 8px 40px rgba(0,0,0,.12);
}}
.auth-logo-name {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem; font-weight: 800; color: {txt};
}}
.auth-logo-sub {{ font-size: 11.5px; color: {muted}; }}
.auth-title {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.5rem; font-weight: 800; color: {txt};
    margin: 18px 0 4px;
}}
.auth-sub {{ font-size: 13.5px; color: {muted}; margin-bottom: 20px; }}

/* ── Topbar brand ── */
.sn-topbar {{
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid {bdr};
}}
.sn-topbar .brand {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem; font-weight: 800; color: {txt};
}}
.sn-topbar .sub {{
    font-size: 11.5px; color: {muted};
}}

/* ── Table ── */
.sn-table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
.sn-table th {{
    text-align: left; padding: 10px 12px;
    font-size: 11.5px; font-weight: 600;
    color: {muted}; text-transform: uppercase; letter-spacing: .05em;
    border-bottom: 1px solid {bdr};
}}
.sn-table td {{
    padding: 11px 12px;
    border-bottom: 1px solid {bdr};
    color: {txt};
}}
.sn-table tr:last-child td {{ border-bottom: none; }}
.sn-table tr:hover td {{ background: {surf2}; }}

/* ── Empty state ── */
.sn-empty {{
    text-align: center; padding: 40px 20px;
    color: {muted};
}}
.sn-empty-ico {{ font-size: 2.5rem; margin-bottom: 10px; }}
.sn-empty-t {{ font-weight: 700; font-size: 1rem; color: {txt}; margin-bottom: 6px; }}

/* ── CTA card ── */
.cta-card {{
    background: linear-gradient(135deg, {acc}, #7c3aed);
    border-radius: 16px;
    padding: 26px;
    color: #fff;
}}
.cta-card h3 {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem; font-weight: 800; margin-bottom: 6px;
}}
.cta-card p {{ font-size: 13px; opacity: .88; margin-bottom: 16px; }}

/* ── Notification item ── */
.notif-item {{
    display: flex; gap: 14px; align-items: flex-start;
    padding: 14px 0;
    border-bottom: 1px solid {bdr};
}}
.notif-item:last-child {{ border-bottom: none; }}
.ni-ico {{
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}}
.ni-success {{ background: rgba(22,163,74,.12); }}
.ni-error   {{ background: rgba(220,38,38,.12); }}
.ni-warning {{ background: rgba(217,119,6,.12); }}
.ni-info    {{ background: {v['--asoft']}; }}
.notif-msg {{ font-size: 13.5px; color: {txt}; }}
.notif-time {{ font-size: 11.5px; color: {muted}; margin-top: 3px; }}
.unread-dot {{
    width: 7px; height: 7px;
    border-radius: 50%;
    background: {acc};
    flex-shrink: 0; margin-top: 6px;
}}

/* ── Avatar ── */
.av {{
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: .9rem; color: #fff;
}}

/* ── Stack tags ── */
.stack-tag {{
    display: inline-block;
    background: {surf2};
    border: 1px solid {bdr};
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 12.5px;
    font-weight: 600;
    color: {txt};
    margin: 3px;
}}

/* ── Pill selector ── */
.part-pill {{
    display: inline-block;
    padding: 5px 14px;
    border-radius: 100px;
    font-size: 12.5px;
    font-weight: 600;
    cursor: pointer;
    background: {surf2};
    border: 1px solid {bdr};
    color: {txt};
    margin: 3px;
}}
.part-pill.active {{
    background: {acc};
    color: #fff;
    border-color: {acc};
}}

/* ── How it works steps ── */
.step-num {{
    font-family: 'Outfit', sans-serif;
    font-size: 1.9rem; font-weight: 800;
    color: {acc}; margin-bottom: 6px;
}}
.step-t {{ font-weight: 700; font-size: .9rem; margin-bottom: 4px; color: {txt}; }}
.step-d {{ font-size: 12.5px; color: {muted}; }}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stDecoration"] {{ display: none; }}
.stDeployButton {{ display: none; }}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Plotly chart renderer  (mirrors renderChart from main.js)
# ═════════════════════════════════════════════════════════════════════════════
def render_chart(pd_data: dict, title: str = "") -> go.Figure:
    t = st.session_state.get("theme", "light")
    tv = THEME_CSS[t]
    dark = t == "dark"; neon = t == "neon"
    grid  = tv["--bdr"]
    label = tv["--muted"]
    acc   = "#60a5fa" if neon else "#2563eb"
    fore  = "#f472b6" if neon else "#d97706"
    bg    = tv["--surf"]

    traces = [
        go.Scatter(
            x=pd_data["hist_dates"], y=pd_data["hist_actual"],
            name="Actual", mode="lines+markers",
            line=dict(color=acc, width=2.5),
            marker=dict(size=5, color=acc),
            hovertemplate="<b>%{y}</b><extra>Actual</extra>"
        ),
        go.Scatter(
            x=pd_data["hist_dates"], y=pd_data["hist_pred"],
            name="Model Fit", mode="lines",
            line=dict(color=acc, width=1.5, dash="dot"),
            opacity=0.45,
            hovertemplate="%{y}<extra>Fit</extra>"
        ),
        go.Scatter(
            x=pd_data["fut_dates"], y=pd_data["fut_vals"],
            name="Forecast", mode="lines+markers",
            line=dict(color=fore, width=2.5),
            marker=dict(size=7, color=fore, symbol="diamond"),
            fill="tozeroy",
            fillcolor="rgba(244,114,182,.06)" if neon else ("rgba(217,119,6,.07)" if dark else "rgba(217,119,6,.06)"),
            hovertemplate="<b>%{y}</b><extra>Forecast</extra>"
        ),
    ]

    shapes = []
    if pd_data.get("hist_dates"):
        shapes.append(dict(
            type="line",
            x0=pd_data["hist_dates"][-1], x1=pd_data["hist_dates"][-1],
            y0=0, y1=1, yref="paper",
            line=dict(color=label, width=1.2, dash="dot")
        ))

    layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,sans-serif", color=label, size=12),
        margin=dict(l=48, r=16, t=20, b=50),
        legend=dict(orientation="h", y=-0.24, font=dict(size=11)),
        xaxis=dict(gridcolor=grid, linecolor=grid, tickfont=dict(size=11), showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=grid, linecolor=grid, tickfont=dict(size=11), showgrid=True, zeroline=False),
        hovermode="x unified",
        shapes=shapes,
        height=340,
        title=dict(text=title, font=dict(size=14, color=tv["--txt"])) if title else None,
    )
    return go.Figure(data=traces, layout=layout)


# ═════════════════════════════════════════════════════════════════════════════
#  Sidebar navigation
# ═════════════════════════════════════════════════════════════════════════════
def _sidebar():
    cu = get_current_user()
    if not cu:
        return

    tv = THEME_CSS[st.session_state.get("theme", "light")]
    nc = unread_notif_count(cu["id"])

    with st.sidebar:
        # Brand
        st.markdown(f"""
<div style="padding:18px 14px 14px;border-bottom:1px solid {tv['--bdr']};margin-bottom:10px">
  <div style="display:flex;align-items:center;gap:10px">
    <div style="font-size:1.8rem">🔵</div>
    <div>
      <div style="font-family:'Outfit',sans-serif;font-size:1rem;font-weight:800;color:#fff">SPREX NOVA</div>
      <div style="font-size:11px;color:{tv['--muted']}">Forecasting the future of parts</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"<div style='font-size:10.5px;font-weight:600;letter-spacing:.08em;color:{tv['--muted']};padding:4px 14px 2px;text-transform:uppercase'>Main</div>", unsafe_allow_html=True)
        page = st.session_state.get("page", "dashboard")

        for label, pg, ico in [
            ("Dashboard", "dashboard", "🏠"),
            ("Upload Data", "upload", "📤"),
            ("Forecast Results", "results_list", "📊"),
        ]:
            active = page == pg
            style = f"background:rgba(255,255,255,.1);color:#fff;" if active else ""
            if st.button(f"{ico}  {label}", key=f"nav_{pg}", use_container_width=True):
                nav(pg)

        st.markdown(f"<div style='font-size:10.5px;font-weight:600;letter-spacing:.08em;color:{tv['--muted']};padding:8px 14px 2px;text-transform:uppercase'>Account</div>", unsafe_allow_html=True)

        notif_label = f"🔔  Notifications{'  (' + str(nc) + ')' if nc > 0 else ''}"
        if st.button(notif_label, key="nav_notifs", use_container_width=True):
            nav("notifications")
        if st.button("👤  Profile", key="nav_profile", use_container_width=True):
            nav("profile")
        if st.button("ℹ️  About", key="nav_about", use_container_width=True):
            nav("about")

        st.markdown(f"<div style='font-size:10.5px;font-weight:600;letter-spacing:.08em;color:{tv['--muted']};padding:8px 14px 2px;text-transform:uppercase'>Session</div>", unsafe_allow_html=True)
        if st.button("🚪  Logout", key="nav_logout", use_container_width=True):
            st.session_state["user_id"] = None
            st.session_state["page"] = "login"
            flash("You've been signed out.", "info")
            st.rerun()

        # Theme switcher
        st.markdown(f"<div style='padding:10px 14px;border-top:1px solid {tv['--bdr']};margin-top:10px'>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:10.5px;font-weight:600;color:{tv['--muted']};margin-bottom:6px;text-transform:uppercase;letter-spacing:.06em'>Theme</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("☀️ Light", key="th_light", use_container_width=True):
                st.session_state["theme"] = "light"; st.rerun()
        with c2:
            if st.button("🌙 Dark", key="th_dark", use_container_width=True):
                st.session_state["theme"] = "dark"; st.rerun()
        with c3:
            if st.button("💜 Neon", key="th_neon", use_container_width=True):
                st.session_state["theme"] = "neon"; st.rerun()

        # User footer
        st.markdown(f"""
<div style="margin-top:14px;padding:12px 14px;border-top:1px solid {tv['--bdr']};display:flex;align-items:center;gap:10px">
  <div style="width:34px;height:34px;border-radius:50%;background:{cu['avatar_color']};display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.9rem;color:#fff;flex-shrink:0">{cu['name'][0].upper()}</div>
  <div>
    <div style="font-size:13px;font-weight:600;color:#fff">{cu['name']}</div>
    <div style="font-size:11px;color:{tv['--muted']}">{cu['email']}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Pages
# ═════════════════════════════════════════════════════════════════════════════

# ── LOGIN ─────────────────────────────────────────────────────────────────────
def page_login():
    tv = THEME_CSS[st.session_state.get("theme", "light")]
    _show_flashes()

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown(f"""
<div class="sn-card">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px">
    <div style="font-size:2rem">🔵</div>
    <div>
      <div class="auth-logo-name">SPREX NOVA</div>
      <div class="auth-logo-sub">Forecasting The Future of Parts</div>
    </div>
  </div>
  <div class="auth-title">Welcome back</div>
  <div class="auth-sub">Sign in to continue to your dashboard</div>
</div>
""", unsafe_allow_html=True)

        with st.form("login_form"):
            email = st.text_input("Email address", placeholder="you@company.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Sign in →", use_container_width=True, type="primary")

        if submitted:
            email = email.strip().lower()
            user = _DB["users"].get(email)
            if user and user["pw_hash"] == _hash(password):
                st.session_state["user_id"] = user["id"]
                flash(f"Welcome back, {user['name'].split()[0]}!", "success")
                nav("dashboard")
            else:
                flash("Invalid email or password.", "error")
                st.rerun()

        st.markdown(f"<div style='text-align:center;margin-top:12px;font-size:13.5px;color:{tv['--muted']}'>Don't have an account? </div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("Create one free →", use_container_width=True):
                nav("signup")
            if st.button("Forgot password?", use_container_width=True):
                nav("forgot_password")


# ── SIGNUP ────────────────────────────────────────────────────────────────────
def page_signup():
    tv = THEME_CSS[st.session_state.get("theme", "light")]
    _show_flashes()

    _, col, _ = st.columns([0.8, 1.4, 0.8])
    with col:
        st.markdown(f"""
<div class="sn-card">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px">
    <div style="font-size:2rem">🔵</div>
    <div>
      <div class="auth-logo-name">SPREX NOVA</div>
      <div class="auth-logo-sub">Create your account</div>
    </div>
  </div>
  <div class="auth-title">Get started</div>
  <div class="auth-sub">Free forever — no credit card needed</div>
</div>
""", unsafe_allow_html=True)

        with st.form("signup_form"):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Full name *", placeholder="Jane Smith")
            with c2:
                company = st.text_input("Company", placeholder="ACME Corp")
            email = st.text_input("Email *", placeholder="you@company.com")
            password = st.text_input("Password *", type="password", placeholder="Min 6 characters")
            confirm = st.text_input("Confirm password *", type="password", placeholder="Repeat password")
            submitted = st.form_submit_button("Create account →", use_container_width=True, type="primary")

        if submitted:
            name = name.strip(); email = email.strip().lower(); company = company.strip()
            if not name or not email or not password:
                flash("Please fill all required fields.", "error")
            elif password != confirm:
                flash("Passwords do not match.", "error")
            elif len(password) < 6:
                flash("Password must be at least 6 characters.", "error")
            elif email in _DB["users"]:
                flash("An account with that email already exists.", "error")
            else:
                uid = _next_id("users")
                colors = ["#5d5fef","#dc2626","#2563eb","#16a34a","#db2777","#d97706","#7c3aed","#0891b2"]
                _DB["users"][email] = {
                    "id": uid, "name": name, "email": email,
                    "pw_hash": _hash(password), "company": company, "role": "Analyst",
                    "avatar_color": colors[int(uid) % len(colors)],
                    "created_at": datetime.utcnow()
                }
                st.session_state["user_id"] = uid
                _add_notif(uid, "success", "Welcome to SPREX NOVA! Upload your first dataset to get started.")
                flash("Account created! Welcome to SPREX NOVA.", "success")
                nav("dashboard")
            st.rerun()

        st.markdown(f"<div style='text-align:center;margin-top:12px;font-size:13.5px;color:{tv['--muted']}'>Already have an account?</div>", unsafe_allow_html=True)
        _, c, _ = st.columns([1, 2, 1])
        with c:
            if st.button("Sign in →", use_container_width=True):
                nav("login")


# ── FORGOT PASSWORD ───────────────────────────────────────────────────────────
def page_forgot_password():
    tv = THEME_CSS[st.session_state.get("theme", "light")]
    _show_flashes()

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown(f"""
<div class="sn-card">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px">
    <div style="font-size:2rem">🔵</div>
    <div>
      <div class="auth-logo-name">SPREX NOVA</div>
      <div class="auth-logo-sub">Password recovery</div>
    </div>
  </div>
  <div class="auth-title">Reset password</div>
  <div class="auth-sub">Enter your email and choose a new password</div>
</div>
""", unsafe_allow_html=True)

        with st.form("forgot_form"):
            email = st.text_input("Email address", placeholder="you@company.com")
            new_pw = st.text_input("New password", type="password", placeholder="Min 6 characters")
            confirm = st.text_input("Confirm new password", type="password", placeholder="Repeat")
            submitted = st.form_submit_button("Reset password", use_container_width=True, type="primary")

        if submitted:
            email = email.strip().lower()
            user = _DB["users"].get(email)
            if not user:
                flash("No account found with that email.", "error")
            elif new_pw != confirm:
                flash("Passwords do not match.", "error")
            elif len(new_pw) < 6:
                flash("Password must be at least 6 characters.", "error")
            else:
                user["pw_hash"] = _hash(new_pw)
                flash("Password updated! Please sign in.", "success")
                nav("login")
            st.rerun()

        _, c, _ = st.columns([1, 2, 1])
        with c:
            if st.button("← Back to login", use_container_width=True):
                nav("login")


# ── DASHBOARD ─────────────────────────────────────────────────────────────────
def page_dashboard():
    cu = get_current_user()
    tv = THEME_CSS[st.session_state.get("theme", "light")]
    _show_flashes()

    uid = cu["id"]
    user_uploads = [u for u in _DB["uploads"].values() if u["user_id"] == uid]
    user_uploads.sort(key=lambda x: x["uploaded_at"], reverse=True)
    total  = len(user_uploads)
    succ   = sum(1 for u in user_uploads if u["status"] == "success")
    total_parts = sum(u["parts_count"] for u in user_uploads if u["status"] == "success")
    recent = user_uploads[:5]

    # Page header
    st.markdown(f"""
<div class="sn-ph">
  <h1>Welcome, <i>{cu['name'].split()[0]}</i> 👋</h1>
  <p>Here's your forecasting overview.</p>
</div>
""", unsafe_allow_html=True)

    # Stat cards
    st.markdown(f"""
<div class="sn-stat-grid">
  <div class="sn-stat">
    <div class="sn-stat-ico">📁</div>
    <div><div class="sn-stat-val">{total}</div><div class="sn-stat-lbl">Total Uploads</div></div>
  </div>
  <div class="sn-stat">
    <div class="sn-stat-ico" style="background:rgba(22,163,74,.1);color:{tv['--green']}">✅</div>
    <div><div class="sn-stat-val">{succ}</div><div class="sn-stat-lbl">Successful</div></div>
  </div>
  <div class="sn-stat">
    <div class="sn-stat-ico" style="background:rgba(2,132,199,.1);color:{tv['--sky']}">🔧</div>
    <div><div class="sn-stat-val">{total_parts}</div><div class="sn-stat-lbl">Parts Analysed</div></div>
  </div>
  <div class="sn-stat">
    <div class="sn-stat-ico" style="background:rgba(217,119,6,.1);color:{tv['--amber']}">🤖</div>
    <div><div class="sn-stat-val">ML</div><div class="sn-stat-lbl">Powered Engine</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

    # CTA + View results
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
<div class="cta-card">
  <div style="font-size:2rem;margin-bottom:10px">⚡</div>
  <h3>Run a New Forecast</h3>
  <p>Upload any spare parts CSV or Excel file and get AI-powered demand forecasts in seconds.</p>
</div>
""", unsafe_allow_html=True)
        if st.button("📤  Upload Dataset →", use_container_width=True, key="dash_upload"):
            nav("upload")

    with c2:
        st.markdown(f"""
<div class="sn-card" style="height:100%">
  <div style="font-size:2rem;margin-bottom:10px">📊</div>
  <h3 style="font-family:'Outfit',sans-serif;font-size:1.05rem;margin-bottom:6px;color:{tv['--txt']}">View Results</h3>
  <p style="font-size:13px;color:{tv['--muted']};margin-bottom:16px">Browse previous forecast results, charts, and demand insights.</p>
</div>
""", unsafe_allow_html=True)
        if st.button("📊  Browse Forecasts →", use_container_width=True, key="dash_results"):
            nav("results_list")

    # Recent uploads table
    st.markdown(f"""
<div class="sn-card">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
    <h2 style="font-size:1rem;color:{tv['--txt']};margin:0">Recent Uploads</h2>
  </div>
""", unsafe_allow_html=True)

    if recent:
        rows = ""
        for u in recent:
            if u["status"] == "success":
                chip = f'<span class="chip chip-green">✅ Success</span>'
            elif u["status"] == "error":
                chip = f'<span class="chip chip-red">❌ Error</span>'
            else:
                chip = f'<span class="chip chip-grey">⏳ Pending</span>'
            rows += f"""
<tr>
  <td><b>{u['filename']}</b></td>
  <td>{u['parts_count']}</td>
  <td>{chip}</td>
  <td style="color:{tv['--muted']};font-size:12.5px">{u['uploaded_at'].strftime('%b %d, %Y')}</td>
</tr>"""
        st.markdown(f"""
<table class="sn-table">
  <thead><tr><th>File</th><th>Parts</th><th>Status</th><th>Date</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
""", unsafe_allow_html=True)

        # View buttons for each successful recent upload
        for u in recent:
            if u["status"] == "success":
                if st.button(f"View → {u['filename']}", key=f"dash_view_{u['id']}"):
                    st.session_state["view_result_id"] = u["id"]
                    nav("results")
    else:
        st.markdown(f"""
<div class="sn-empty">
  <div class="sn-empty-ico">📂</div>
  <div class="sn-empty-t">No uploads yet</div>
  <p style="font-size:13px">Upload your first dataset to get started.</p>
</div>
""", unsafe_allow_html=True)
        if st.button("Upload Now →", key="dash_upload2", type="primary"):
            nav("upload")

    st.markdown("</div>", unsafe_allow_html=True)


# ── UPLOAD ────────────────────────────────────────────────────────────────────
def page_upload():
    cu = get_current_user()
    tv = THEME_CSS[st.session_state.get("theme", "light")]
    _show_flashes()

    uid = cu["id"]

    st.markdown(f"""
<div class="sn-ph">
  <h1>Upload Dataset</h1>
  <p>Drop any CSV or Excel file — SPREX NOVA adapts to your data automatically.</p>
</div>
""", unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 0.8])

    with c1:
        st.markdown(f'<div class="sn-card">', unsafe_allow_html=True)
        st.markdown(f"**Select File**")
        st.caption("CSV · XLSX · XLS · Max 32 MB")

        uploaded_file = st.file_uploader(
            "Drop your file here or click to browse",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed"
        )

        steps = st.selectbox(
            "Forecast horizon",
            options=[3, 6, 12, 24],
            index=1,
            format_func=lambda x: f"{x} periods ahead"
        )
        st.caption("How many time steps into the future to forecast.")

        run_btn = st.button("⚡ Run Forecast", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if run_btn:
            if not uploaded_file:
                flash("No file selected.", "error")
                st.rerun()
            else:
                filename = uploaded_file.name
                ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
                if ext not in ("csv", "xlsx", "xls"):
                    flash("Only CSV, XLSX, and XLS files are supported.", "error")
                    st.rerun()
                else:
                    with st.spinner("Running ML forecast…"):
                        try:
                            raw = uploaded_file.read()
                            if ext == "csv":
                                df = pd.read_csv(io.BytesIO(raw))
                            else:
                                df = pd.read_excel(io.BytesIO(raw), engine="openpyxl" if HAS_OPENPYXL else None)

                            if df.empty or len(df.columns) < 2:
                                raise ValueError("Dataset must have at least 2 columns.")

                            result = run_forecast(df, steps=int(steps))
                            parts_count = len(result.get("parts", {}))
                            status = "success"
                            _add_notif(uid, "success", f"Forecast complete for '{filename}' — {parts_count} part(s) analysed.")
                        except Exception as e:
                            result = {"error": str(e)}
                            parts_count = 0
                            status = "error"
                            _add_notif(uid, "error", f"Forecast failed for '{filename}': {e}")

                    rec_id = _next_id("uploads")
                    _DB["uploads"][rec_id] = {
                        "id": rec_id, "user_id": uid, "filename": filename,
                        "status": status, "parts_count": parts_count,
                        "uploaded_at": datetime.utcnow(),
                        "result_json": json.dumps(result, default=str)
                    }

                    if status == "success":
                        flash(f"Forecast complete! {parts_count} part(s) analysed.", "success")
                        st.session_state["view_result_id"] = rec_id
                        nav("results")
                    else:
                        flash(f"Forecast failed: {result.get('error')}", "error")
                        st.rerun()

    with c2:
        st.markdown(f"""
<div class="sn-card">
  <h3 style="font-size:.9rem;margin-bottom:12px;color:{tv['--txt']}">📋 Supported column names</h3>
  <p style="font-size:12.5px;color:{tv['--muted']};margin-bottom:10px">The engine auto-detects these patterns:</p>
  <div style="display:flex;gap:9px;align-items:flex-start;margin-bottom:9px">
    <span>📅</span>
    <div><div style="font-weight:600;font-size:12.5px;color:{tv['--txt']}">Date / Period</div><div style="font-size:11.5px;color:{tv['--muted']}">date, month, period, time…</div></div>
  </div>
  <div style="display:flex;gap:9px;align-items:flex-start;margin-bottom:9px">
    <span>🔢</span>
    <div><div style="font-weight:600;font-size:12.5px;color:{tv['--txt']}">Demand / Quantity</div><div style="font-size:11.5px;color:{tv['--muted']}">demand, qty, sales, usage…</div></div>
  </div>
  <div style="display:flex;gap:9px;align-items:flex-start;margin-bottom:9px">
    <span>🔧</span>
    <div><div style="font-weight:600;font-size:12.5px;color:{tv['--txt']}">Part / SKU</div><div style="font-size:11.5px;color:{tv['--muted']}">part, item, sku, product…</div></div>
  </div>
  <div style="margin-top:12px;padding:9px 12px;background:{tv['--surf2']};border-radius:10px;font-size:12px;color:{tv['--muted']}">
    💡 No renaming needed — works with any column structure.
  </div>
</div>

<div class="sn-card" style="margin-top:14px">
  <h3 style="font-size:.9rem;margin-bottom:12px;color:{tv['--txt']}">🤖 Models used</h3>
  <div style="display:flex;gap:8px;align-items:flex-start;margin-bottom:8px">
    <div style="width:7px;height:7px;border-radius:50%;background:{tv['--accent']};flex-shrink:0;margin-top:4px"></div>
    <div><div style="font-weight:600;font-size:12.5px;color:{tv['--txt']}">Random Forest</div><div style="font-size:11.5px;color:{tv['--muted']}">≥ 8 records per part — high accuracy</div></div>
  </div>
  <div style="display:flex;gap:8px;align-items:flex-start;margin-bottom:8px">
    <div style="width:7px;height:7px;border-radius:50%;background:{tv['--accent']};flex-shrink:0;margin-top:4px"></div>
    <div><div style="font-weight:600;font-size:12.5px;color:{tv['--txt']}">Linear Regression</div><div style="font-size:11.5px;color:{tv['--muted']}">Smaller datasets — fast & stable</div></div>
  </div>
  <div style="display:flex;gap:8px;align-items:flex-start;margin-bottom:8px">
    <div style="width:7px;height:7px;border-radius:50%;background:{tv['--accent']};flex-shrink:0;margin-top:4px"></div>
    <div><div style="font-weight:600;font-size:12.5px;color:{tv['--txt']}">Auto-selection</div><div style="font-size:11.5px;color:{tv['--muted']}">Engine picks the best model automatically</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Upload history
    user_uploads = sorted(
        [u for u in _DB["uploads"].values() if u["user_id"] == uid],
        key=lambda x: x["uploaded_at"], reverse=True
    )

    if user_uploads:
        st.markdown(f'<div class="sn-card" style="margin-top:20px"><h2 style="font-size:.95rem;margin-bottom:16px;color:{tv["--txt"]}">Upload History</h2>', unsafe_allow_html=True)
        rows = ""
        for u in user_uploads:
            chip = f'<span class="chip chip-green">✅ Success</span>' if u["status"] == "success" else f'<span class="chip chip-red">❌ Error</span>'
            rows += f"""
<tr>
  <td style="font-weight:500">{u['filename']}</td>
  <td>{u['parts_count']}</td>
  <td>{chip}</td>
  <td style="font-size:12.5px;color:{tv['--muted']}">{u['uploaded_at'].strftime('%b %d, %Y %H:%M')}</td>
</tr>"""
        st.markdown(f'<table class="sn-table"><thead><tr><th>File</th><th>Parts</th><th>Status</th><th>Date</th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        for u in user_uploads:
            if u["status"] == "success":
                if st.button(f"View → {u['filename']}", key=f"hist_view_{u['id']}"):
                    st.session_state["view_result_id"] = u["id"]
                    nav("results")


# ── RESULTS LIST ──────────────────────────────────────────────────────────────
def page_results_list():
    cu = get_current_user()
    tv = THEME_CSS[st.session_state.get("theme", "light")]
    _show_flashes()

    uid = cu["id"]
    recs = sorted(
        [u for u in _DB["uploads"].values()
         if u["user_id"] == uid and u["status"] == "success"],
        key=lambda x: x["uploaded_at"], reverse=True
    )

    st.markdown(f"""
<div class="sn-ph"><h1>Forecast Results</h1><p>All your completed forecast jobs.</p></div>
""", unsafe_allow_html=True)

    st.markdown('<div class="sn-card">', unsafe_allow_html=True)

    if recs:
        rows = ""
        for r in recs:
            rows += f"""
<tr>
  <td style="font-weight:600">{r['filename']}</td>
  <td><span class="chip chip-purple">{r['parts_count']} part{'s' if r['parts_count']!=1 else ''}</span></td>
  <td style="font-size:12.5px;color:{tv['--muted']}">{r['uploaded_at'].strftime('%b %d, %Y %H:%M')}</td>
</tr>"""
        st.markdown(f'<table class="sn-table"><thead><tr><th>Dataset</th><th>Parts Forecasted</th><th>Date</th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("**Open a forecast:**")
        for r in recs:
            if st.button(f"View → {r['filename']}  ({r['parts_count']} parts)", key=f"rl_{r['id']}"):
                st.session_state["view_result_id"] = r["id"]
                nav("results")
    else:
        st.markdown(f"""
<div class="sn-empty">
  <div class="sn-empty-ico">📊</div>
  <div class="sn-empty-t">No results yet</div>
  <p style="font-size:13px">Upload a dataset to generate your first forecast.</p>
</div>
""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("Upload Dataset →", type="primary"):
            nav("upload")


# ── RESULTS (detail) ──────────────────────────────────────────────────────────
def page_results():
    cu = get_current_user()
    tv = THEME_CSS[st.session_state.get("theme", "light")]
    _show_flashes()

    uid = cu["id"]
    rec_id = st.session_state.get("view_result_id")

    if not rec_id:
        flash("No result selected.", "warning")
        nav("results_list")
        return

    rec = _DB["uploads"].get(str(rec_id))
    if not rec or rec["user_id"] != uid:
        flash("Result not found.", "error")
        nav("results_list")
        return

    data = json.loads(rec["result_json"])

    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"""
<div class="sn-ph">
  <h1>{rec['filename']}</h1>
  <p>{rec['parts_count']} part{'s' if rec['parts_count']!=1 else ''} · {rec['uploaded_at'].strftime('%B %d, %Y %H:%M')}</p>
</div>
""", unsafe_allow_html=True)
    with col2:
        if st.button("← All Results", key="back_results"):
            nav("results_list")

    if data.get("error"):
        st.error(f"❌ Forecast Error: {data['error']}")
        return

    parts = data.get("parts", {})
    warns = data.get("warnings", [])

    if warns:
        for w in warns:
            st.warning(w)

    # Summary stats
    n_high  = sum(1 for p in parts.values() if p.get("status") == "High Demand Expected")
    n_stock = sum(1 for p in parts.values() if p.get("status") == "Risk of Stockout")
    n_over  = sum(1 for p in parts.values() if p.get("status") == "Overstock Risk")
    n_buy   = sum(1 for p in parts.values() if p.get("status") == "Purchase Recommended")

    st.markdown(f"""
<div class="sn-stat-grid">
  <div class="sn-stat">
    <div class="sn-stat-ico">🔧</div>
    <div><div class="sn-stat-val">{rec['parts_count']}</div><div class="sn-stat-lbl">Parts Forecasted</div></div>
  </div>
  <div class="sn-stat">
    <div class="sn-stat-ico" style="background:rgba(220,38,38,.1);color:{tv['--red']}">📈</div>
    <div><div class="sn-stat-val">{n_high}</div><div class="sn-stat-lbl">High Demand</div></div>
  </div>
  <div class="sn-stat">
    <div class="sn-stat-ico" style="background:rgba(217,119,6,.1);color:{tv['--amber']}">⚠️</div>
    <div><div class="sn-stat-val">{n_stock}</div><div class="sn-stat-lbl">Stockout Risk</div></div>
  </div>
  <div class="sn-stat">
    <div class="sn-stat-ico" style="background:rgba(22,163,74,.1);color:{tv['--green']}">🛒</div>
    <div><div class="sn-stat-val">{n_buy}</div><div class="sn-stat-lbl">Purchase Alerts</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

    plist = [p for p in parts.keys() if not parts[p].get("error")]
    if not plist:
        st.info("No valid parts to display.")
        return

    # Part selector
    st.markdown(f'<div class="sn-card"><h2 style="font-size:.9rem;margin-bottom:10px;color:{tv["--txt"]}">Select Part</h2>', unsafe_allow_html=True)
    selected_part = st.selectbox("Choose a part to view its chart", plist, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    pd_data = parts.get(selected_part, {})

    # Status chip mapping
    STATUS_MAP = {
        "High Demand Expected": ("chip-red",   "📈"),
        "Risk of Stockout":     ("chip-amber",  "⚠️"),
        "Overstock Risk":       ("chip-sky",    "📦"),
        "Purchase Recommended": ("chip-green",  "🛒"),
        "Demand Stable":        ("chip-blue",   "✅"),
    }
    s = pd_data.get("status", "Demand Stable")
    cls, ico = STATUS_MAP.get(s, ("chip-blue", "✅"))

    st.markdown(f"""
<div class="sn-card">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:16px">
    <div>
      <h2 style="font-size:1.1rem;margin-bottom:6px;color:{tv['--txt']}">{selected_part}</h2>
      <span class="chip {cls}">{ico} {s}</span>
      <span style="font-size:12px;color:{tv['--muted']};margin-left:10px">🤖 {pd_data.get('model','')}</span>
      <span style="font-size:12px;color:{tv['--muted']};margin-left:10px">MAE: {pd_data.get('mae','')}</span>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <span style="font-size:12px;color:{tv['--muted']}">▬ Actual</span>
      <span style="font-size:12px;color:{tv['--muted']}">▬ Forecast</span>
    </div>
  </div>
""", unsafe_allow_html=True)

    fig = render_chart(pd_data)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Forecast table
    st.markdown(f"<h3 style='font-size:.85rem;color:{tv['--muted']};margin:16px 0 10px'>Forecast Values</h3>", unsafe_allow_html=True)
    fut_dates = pd_data.get("fut_dates", [])
    fut_vals  = pd_data.get("fut_vals", [])
    if fut_dates:
        rows = ""
        for i, (d, v) in enumerate(zip(fut_dates, fut_vals)):
            rows += f"<tr><td>{d}</td><td style='font-weight:600'>{v}</td><td><span class='chip {cls}'>{ico} {s}</span></td></tr>"
        st.markdown(f'<table class="sn-table"><thead><tr><th>Period</th><th>Forecasted Demand</th><th>Insight</th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Full summary table
    st.markdown(f'<div class="sn-card"><h2 style="font-size:.95rem;margin-bottom:14px;color:{tv["--txt"]}">All Parts Summary</h2>', unsafe_allow_html=True)
    rows = ""
    for pname, pd_item in parts.items():
        if pd_item.get("error"):
            continue
        si = pd_item.get("status", "Demand Stable")
        cl, ic = STATUS_MAP.get(si, ("chip-blue", "✅"))
        rows += f"""
<tr>
  <td style="font-weight:600">{pname}</td>
  <td style="font-size:12.5px;color:{tv['--muted']}">{pd_item.get('model','')}</td>
  <td>{pd_item.get('avg_hist','')}</td>
  <td style="font-weight:600">{pd_item.get('avg_fore','')}</td>
  <td style="color:{tv['--muted']}">{pd_item.get('mae','')}</td>
  <td><span class="chip {cl}">{ic} {si}</span></td>
</tr>"""
    st.markdown(f"""
<table class="sn-table">
  <thead><tr><th>Part</th><th>Model</th><th>Avg Historical</th><th>Avg Forecast</th><th>MAE</th><th>Status</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── NOTIFICATIONS ─────────────────────────────────────────────────────────────
def page_notifications():
    cu = get_current_user()
    tv = THEME_CSS[st.session_state.get("theme", "light")]
    _show_flashes()

    uid = cu["id"]
    notifs = sorted(
        [n for n in _DB["notifs"].values() if n["user_id"] == uid],
        key=lambda x: x["created_at"], reverse=True
    )
    for n in notifs:
        n["read"] = True

    st.markdown(f"""
<div class="sn-ph"><h1>Notifications</h1><p>Your recent activity and alerts.</p></div>
""", unsafe_allow_html=True)

    st.markdown('<div class="sn-card">', unsafe_allow_html=True)

    if notifs:
        for n in notifs:
            nt = n.get("ntype", "info")
            ico_map = {"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}
            bg_map  = {"success": "ni-success", "error": "ni-error", "warning": "ni-warning", "info": "ni-info"}
            ico = ico_map.get(nt, "ℹ️")
            bg  = bg_map.get(nt, "ni-info")
            dot = '<div class="unread-dot"></div>' if not n.get("read") else ""
            st.markdown(f"""
<div class="notif-item">
  <div class="ni-ico {bg}">{ico}</div>
  <div style="flex:1">
    <div class="notif-msg">{n['message']}</div>
    <div class="notif-time">{n['created_at'].strftime('%b %d, %Y at %H:%M')}</div>
  </div>
  {dot}
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="sn-empty">
  <div class="sn-empty-ico">🔔</div>
  <div class="sn-empty-t">All caught up!</div>
  <p style="font-size:13px">Notifications appear here after you run forecasts.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ── PROFILE ───────────────────────────────────────────────────────────────────
def page_profile():
    cu = get_current_user()
    tv = THEME_CSS[st.session_state.get("theme", "light")]
    _show_flashes()

    # Get live reference from DB
    user = next((u for u in _DB["users"].values() if u["id"] == cu["id"]), None)
    if not user:
        nav("login")
        return

    st.markdown(f"""
<div class="sn-ph"><h1>My Profile</h1><p>Manage your account info and security.</p></div>
""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""
<div class="sn-card">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:22px;padding-bottom:20px;border-bottom:1px solid {tv['--bdr']}">
    <div class="av" style="width:58px;height:58px;font-size:1.5rem;background:{user['avatar_color']}">{user['name'][0].upper()}</div>
    <div>
      <h2 style="font-size:1.1rem;margin-bottom:2px;color:{tv['--txt']}">{user['name']}</h2>
      <div style="font-size:12.5px;color:{tv['--muted']}">{user['email']}</div>
      <div style="font-size:11.5px;color:{tv['--muted']};margin-top:2px">Member since {user['created_at'].strftime('%B %Y')}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        with st.form("profile_info_form"):
            new_name    = st.text_input("Full name", value=user["name"])
            st.text_input("Email (read-only)", value=user["email"], disabled=True)
            ci, ri = st.columns(2)
            with ci:
                new_company = st.text_input("Company", value=user.get("company") or "", placeholder="Your company")
            with ri:
                new_role = st.text_input("Role", value=user.get("role") or "Analyst")

            COLORS = ["#5d5fef","#dc2626","#2563eb","#16a34a","#db2777","#d97706","#7c3aed","#0891b2"]
            color_labels = ["Indigo","Red","Blue","Green","Pink","Amber","Purple","Cyan"]
            cur_idx = COLORS.index(user["avatar_color"]) if user["avatar_color"] in COLORS else 0
            new_color_idx = st.selectbox(
                "Avatar color",
                range(len(COLORS)),
                index=cur_idx,
                format_func=lambda i: f"{color_labels[i]}"
            )

            saved = st.form_submit_button("Save changes", type="primary")

        if saved:
            user["name"]         = new_name.strip() or user["name"]
            user["company"]      = new_company.strip()
            user["role"]         = new_role.strip() or "Analyst"
            user["avatar_color"] = COLORS[new_color_idx]
            flash("Profile updated.", "success")
            st.rerun()

    with c2:
        st.markdown(f'<div class="sn-card"><h2 style="font-size:.95rem;margin-bottom:18px;color:{tv["--txt"]}">🔒 Change Password</h2>', unsafe_allow_html=True)

        with st.form("profile_pw_form"):
            current_pw = st.text_input("Current password", type="password", placeholder="Your current password")
            new_pw     = st.text_input("New password", type="password", placeholder="Min 6 characters")
            confirm_pw = st.text_input("Confirm new password", type="password", placeholder="Repeat")
            pw_saved   = st.form_submit_button("Update password", type="primary")

        if pw_saved:
            if user["pw_hash"] != _hash(current_pw):
                flash("Current password is incorrect.", "error")
            elif new_pw != confirm_pw:
                flash("New passwords do not match.", "error")
            elif len(new_pw) < 6:
                flash("Password must be at least 6 characters.", "error")
            else:
                user["pw_hash"] = _hash(new_pw)
                flash("Password updated successfully.", "success")
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # Account info
        st.markdown(f'<div class="sn-card">', unsafe_allow_html=True)
        info_rows = [
            ("Role", user.get("role") or "Analyst"),
            ("Company", user.get("company") or "—"),
            ("Joined", user["created_at"].strftime("%B %d, %Y")),
        ]
        rows = "".join(
            f"<tr><td style='color:{tv['--muted']}'>{lbl}</td><td style='font-weight:600;color:{tv['--txt']}'>{val}</td></tr>"
            for lbl, val in info_rows
        )
        st.markdown(f'<table class="sn-table"><tbody>{rows}</tbody></table>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state["user_id"] = None
            st.session_state["page"] = "login"
            flash("You've been signed out.", "info")
            st.rerun()


# ── ABOUT ─────────────────────────────────────────────────────────────────────
def page_about():
    tv = THEME_CSS[st.session_state.get("theme", "light")]
    _show_flashes()

    st.markdown(f"""
<div class="about-hero">
  <div style="font-size:3rem;margin-bottom:12px">🔵</div>
  <h1>SPREX NOVA</h1>
  <p>Forecasting The Future of Parts — an adaptive ML platform that works with <em>any</em> spare parts dataset and delivers accurate demand predictions.</p>
</div>
""", unsafe_allow_html=True)

    FEATURES = [
        ("⇰","Adaptive ML Engine","Auto-detects columns and trains on your exact dataset. No hard-coded parts — Random Forest and Linear Regression selected automatically."),
        ("⇰","Interactive Charts","Plotly-powered charts showing actual demand vs model fit vs future forecast — with hover details and clean legends."),
        ("⇰","Zero Config","Upload any CSV or Excel file and forecasting begins immediately. Column names are auto-detected."),
        ("⇰","Secure","Hashed passwords, session isolation, safe file handling, and input validation throughout."),
        ("⇰","Three Themes","Switch between Light, Dark, and Neon modes. Theme persists across sessions."),
        ("⇰","Fully Responsive","Clean experience on desktop and mobile with smooth animated transitions."),
        ("⇰","Smart Alerts","High Demand Expected · Risk of Stockout · Overstock Risk · Purchase Recommended · Demand Stable."),
        ("⇰","Full History","Every forecast is saved so you can compare results across different datasets over time."),
    ]

    feat_html = '<div class="feat-grid">'
    for ico, title, desc in FEATURES:
        feat_html += f"""
<div class="feat">
  <div class="feat-ico">{ico}</div>
  <div class="feat-t">{title}</div>
  <p class="feat-d">{desc}</p>
</div>"""
    feat_html += "</div>"
    st.markdown(feat_html, unsafe_allow_html=True)

    # How it works
    st.markdown(f'<div class="sn-card"><h2 style="font-size:.95rem;margin-bottom:16px;color:{tv["--txt"]}">How it works</h2>', unsafe_allow_html=True)
    steps = [("01","Upload","Any CSV or Excel file with spare parts data."),("02","Detect","Columns auto-identified — dates, quantities, part names."),("03","Train","ML model trains per part on your data only."),("04","Forecast","Future demand predicted for each unique part."),("05","Analyse","Charts, tables, and status labels guide your decisions.")]
    cols = st.columns(len(steps))
    for col, (n, t, d) in zip(cols, steps):
        with col:
            st.markdown(f"""
<div style="text-align:center;padding:14px">
  <div class="step-num">{n}</div>
  <div class="step-t">{t}</div>
  <div class="step-d">{d}</div>
</div>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Stack
    STACK = ['Python 3','Streamlit','scikit-learn','pandas','NumPy','Plotly','openpyxl']
    tags = " ".join(f'<span class="stack-tag">{t}</span>' for t in STACK)
    st.markdown(f'<div class="sn-card"><h2 style="font-size:.95rem;margin-bottom:14px;color:{tv["--txt"]}">Stack</h2><div>{tags}</div></div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  Router
# ═════════════════════════════════════════════════════════════════════════════
def main():
    _inject_css()

    cu = get_current_user()
    page = st.session_state.get("page", "login")

    # Redirect unauthenticated users
    if not cu and page not in ("login", "signup", "forgot_password"):
        page = "login"
        st.session_state["page"] = "login"

    # Redirect authenticated users away from auth pages
    if cu and page in ("login", "signup", "forgot_password"):
        page = "dashboard"
        st.session_state["page"] = "dashboard"

    # Show sidebar only for authenticated pages
    if cu and page not in ("login", "signup", "forgot_password"):
        _sidebar()

    # Route
    if page == "login":
        page_login()
    elif page == "signup":
        page_signup()
    elif page == "forgot_password":
        page_forgot_password()
    elif page == "dashboard":
        page_dashboard()
    elif page == "upload":
        page_upload()
    elif page == "results_list":
        page_results_list()
    elif page == "results":
        page_results()
    elif page == "notifications":
        page_notifications()
    elif page == "profile":
        page_profile()
    elif page == "about":
        page_about()
    else:
        nav("dashboard")


if __name__ == "__main__":
    main()