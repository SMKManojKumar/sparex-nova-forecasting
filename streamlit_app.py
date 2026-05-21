"""
SPREX NOVA — Streamlit App
Converted from Flask backend (Vercel-compatible) to Streamlit
All ML logic, password hashing, DB structure, and business logic preserved exactly.
"""

import os, json, io, hashlib, secrets, re
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np

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
#  Streamlit page config
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SPREX NOVA",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═════════════════════════════════════════════════════════════════════════════
#  In-memory "database" stored in session_state
# ═════════════════════════════════════════════════════════════════════════════
if "_DB" not in st.session_state:
    st.session_state._DB = {"users": {}, "uploads": {}, "notifs": {}}

_DB = st.session_state._DB

# ─── simple counter ids ──────────────────────────────────────────────────────
def _next_id(table: str) -> str:
    existing = [int(k) for k in _DB[table] if k.isdigit()]
    return str(max(existing, default=0) + 1)

# ─── password hashing ────────────────────────────────────────────────────────
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# ═════════════════════════════════════════════════════════════════════════════
#  Session helpers
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
#  ML Forecast engine  (preserved exactly from original)
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
        # Simple linear extrapolation fallback
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

    # coerce qty to numeric
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)

    # build date labels
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

        # date labels for historical
        if date_col in grp.columns:
            try:
                hist_dates = [str(d.date()) for d in grp[date_col]]
            except Exception:
                hist_dates = [str(v) for v in grp[date_col]]
        else:
            hist_dates = [f"P{i+1}" for i in range(len(vals))]

        # forecast date labels
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
#  UI — Auth pages
# ═════════════════════════════════════════════════════════════════════════════
def page_login():
    st.title("🚀 SPREX NOVA")
    st.subheader("Sign In")
    with st.form("login_form"):
        email = st.text_input("Email").strip().lower()
        pw    = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign In")
    if submitted:
        user = _DB["users"].get(email)
        if user and user["pw_hash"] == _hash(pw):
            st.session_state["user_id"] = user["id"]
            st.session_state["page"] = "dashboard"
            st.success(f"Welcome back, {user['name'].split()[0]}!")
            st.rerun()
        else:
            st.error("Invalid email or password.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Create Account"):
            st.session_state["page"] = "signup"
            st.rerun()
    with col2:
        if st.button("Forgot Password?"):
            st.session_state["page"] = "forgot_password"
            st.rerun()


def page_signup():
    st.title("🚀 SPREX NOVA")
    st.subheader("Create Account")
    with st.form("signup_form"):
        name    = st.text_input("Full Name *").strip()
        email   = st.text_input("Email *").strip().lower()
        company = st.text_input("Company").strip()
        pw      = st.text_input("Password *", type="password")
        confirm = st.text_input("Confirm Password *", type="password")
        submitted = st.form_submit_button("Create Account")
    if submitted:
        if not name or not email or not pw:
            st.error("Please fill all required fields.")
        elif pw != confirm:
            st.error("Passwords do not match.")
        elif len(pw) < 6:
            st.error("Password must be at least 6 characters.")
        elif email in _DB["users"]:
            st.error("An account with that email already exists.")
        else:
            uid = _next_id("users")
            colors = ["#5d5fef","#dc2626","#2563eb","#16a34a","#db2777","#d97706","#7c3aed","#0891b2"]
            _DB["users"][email] = {
                "id": uid, "name": name, "email": email,
                "pw_hash": _hash(pw), "company": company, "role": "Analyst",
                "avatar_color": colors[int(uid) % len(colors)],
                "created_at": datetime.utcnow()
            }
            st.session_state["user_id"] = uid
            _add_notif(uid, "success", "Welcome to SPREX NOVA! Upload your first dataset to get started.")
            st.session_state["page"] = "dashboard"
            st.success("Account created! Welcome to SPREX NOVA.")
            st.rerun()

    if st.button("← Back to Sign In"):
        st.session_state["page"] = "login"
        st.rerun()


def page_forgot_password():
    st.title("🚀 SPREX NOVA")
    st.subheader("Reset Password")
    with st.form("forgot_form"):
        email   = st.text_input("Email").strip().lower()
        new_pw  = st.text_input("New Password", type="password")
        confirm = st.text_input("Confirm New Password", type="password")
        submitted = st.form_submit_button("Update Password")
    if submitted:
        user = _DB["users"].get(email)
        if not user:
            st.error("No account found with that email.")
        elif new_pw != confirm:
            st.error("Passwords do not match.")
        elif len(new_pw) < 6:
            st.error("Password must be at least 6 characters.")
        else:
            user["pw_hash"] = _hash(new_pw)
            st.success("Password updated! Please sign in.")
            st.session_state["page"] = "login"
            st.rerun()

    if st.button("← Back to Sign In"):
        st.session_state["page"] = "login"
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  UI — App pages (require login)
# ═════════════════════════════════════════════════════════════════════════════
def sidebar_nav(cu):
    nc = unread_notif_count(cu["id"])
    st.sidebar.markdown(f"### 🚀 SPREX NOVA")
    st.sidebar.markdown(f"**{cu['name']}**  \n{cu['email']}")
    st.sidebar.markdown("---")
    pages = {
        "dashboard": "📊 Dashboard",
        "upload": "📁 Upload",
        "results_list": "📈 Results",
        "notifications": f"🔔 Notifications {'🔴' if nc > 0 else ''}",
        "profile": "👤 Profile",
        "about": "ℹ️ About",
    }
    for key, label in pages.items():
        if st.sidebar.button(label, key=f"nav_{key}"):
            st.session_state["page"] = key
            st.rerun()
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Sign Out"):
        st.session_state.clear()
        st.session_state["page"] = "login"
        st.rerun()


def page_dashboard(cu):
    uid = cu["id"]
    user_uploads = [u for u in _DB["uploads"].values() if u["user_id"] == uid]
    user_uploads.sort(key=lambda x: x["uploaded_at"], reverse=True)
    total       = len(user_uploads)
    succ        = sum(1 for u in user_uploads if u["status"] == "success")
    total_parts = sum(u["parts_count"] for u in user_uploads if u["status"] == "success")
    recent      = user_uploads[:5]

    st.title("📊 Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Uploads", total)
    c2.metric("Successful Forecasts", succ)
    c3.metric("Total Parts Analysed", total_parts)

    st.subheader("Recent Uploads")
    if recent:
        rows = []
        for u in recent:
            rows.append({
                "Filename": u["filename"],
                "Status": u["status"].capitalize(),
                "Parts": u["parts_count"],
                "Uploaded At": u["uploaded_at"].strftime("%Y-%m-%d %H:%M"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No uploads yet. Go to Upload to get started.")


def page_upload(cu):
    uid = cu["id"]
    st.title("📁 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV, XLSX, or XLS file",
        type=["csv", "xlsx", "xls"]
    )
    steps = st.slider("Forecast periods ahead", min_value=1, max_value=24, value=6)

    if st.button("Run Forecast", disabled=uploaded_file is None):
        filename = uploaded_file.name
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ("csv", "xlsx", "xls"):
            st.error("Only CSV, XLSX, and XLS files are supported.")
        else:
            try:
                raw = uploaded_file.read()
                if ext == "csv":
                    df = pd.read_csv(io.BytesIO(raw))
                else:
                    df = pd.read_excel(io.BytesIO(raw), engine="openpyxl" if HAS_OPENPYXL else None)

                if df.empty or len(df.columns) < 2:
                    raise ValueError("Dataset must have at least 2 columns.")

                with st.spinner("Running forecast…"):
                    result = run_forecast(df, steps=steps)
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
                st.success(f"Forecast complete! {parts_count} part(s) analysed.")
                st.session_state["view_result_id"] = rec_id
                st.session_state["page"] = "results"
                st.rerun()
            else:
                st.error(f"Forecast failed: {result.get('error')}")

    st.subheader("Your Uploads")
    user_uploads = sorted(
        [u for u in _DB["uploads"].values() if u["user_id"] == uid],
        key=lambda x: x["uploaded_at"], reverse=True
    )
    if user_uploads:
        rows = []
        for u in user_uploads:
            rows.append({
                "ID": u["id"],
                "Filename": u["filename"],
                "Status": u["status"].capitalize(),
                "Parts": u["parts_count"],
                "Uploaded At": u["uploaded_at"].strftime("%Y-%m-%d %H:%M"),
            })
        df_uploads = pd.DataFrame(rows)
        st.dataframe(df_uploads, use_container_width=True)
    else:
        st.info("No uploads yet.")


def page_results_list(cu):
    uid = cu["id"]
    st.title("📈 Forecast Results")
    recs = sorted(
        [u for u in _DB["uploads"].values()
         if u["user_id"] == uid and u["status"] == "success"],
        key=lambda x: x["uploaded_at"], reverse=True
    )
    if not recs:
        st.info("No successful forecasts yet.")
        return

    for rec in recs:
        with st.expander(f"📄 {rec['filename']}  —  {rec['uploaded_at'].strftime('%Y-%m-%d %H:%M')}  ({rec['parts_count']} parts)"):
            if st.button(f"View Results", key=f"view_{rec['id']}"):
                st.session_state["view_result_id"] = rec["id"]
                st.session_state["page"] = "results"
                st.rerun()


def page_results(cu):
    uid = cu["id"]
    rec_id = st.session_state.get("view_result_id")
    rec = _DB["uploads"].get(str(rec_id)) if rec_id else None

    if not rec or rec["user_id"] != uid:
        st.error("Result not found.")
        if st.button("← Back to Results"):
            st.session_state["page"] = "results_list"
            st.rerun()
        return

    data = json.loads(rec["result_json"])

    st.title(f"📈 Forecast: {rec['filename']}")
    st.caption(f"Uploaded: {rec['uploaded_at'].strftime('%Y-%m-%d %H:%M')}")

    if st.button("← Back to Results List"):
        st.session_state["page"] = "results_list"
        st.rerun()

    warnings = data.get("warnings", [])
    if warnings:
        for w in warnings:
            st.warning(w)

    parts = data.get("parts", {})
    if not parts:
        st.info("No parts found in results.")
        return

    # Summary table
    summary_rows = []
    for pname, pdata in parts.items():
        if "error" in pdata:
            summary_rows.append({"Part": pname, "Status": "Error", "Avg Historical": "-", "Avg Forecast": "-", "MAE": "-", "Model": "-"})
        else:
            summary_rows.append({
                "Part": pname,
                "Status": pdata["status"],
                "Avg Historical": pdata["avg_hist"],
                "Avg Forecast": pdata["avg_fore"],
                "MAE": pdata["mae"],
                "Model": pdata["model"],
            })
    st.subheader("Summary")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    # Per-part detail
    st.subheader("Part Details")
    for pname, pdata in parts.items():
        with st.expander(f"🔩 {pname}"):
            if "error" in pdata:
                st.error(pdata["error"])
                continue

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Historical", pdata["avg_hist"])
            col2.metric("Avg Forecast", pdata["avg_fore"])
            col3.metric("MAE", pdata["mae"])
            col4.metric("Model", pdata["model"])

            status = pdata["status"]
            status_color = {
                "High Demand Expected": "🔴",
                "Overstock Risk": "🟠",
                "Risk of Stockout": "🟡",
                "Purchase Recommended": "🔵",
                "Demand Stable": "🟢",
            }.get(status, "⚪")
            st.markdown(f"**Status:** {status_color} {status}")

            # Chart: historical actual vs predicted
            hist_df = pd.DataFrame({
                "Date": pdata["hist_dates"],
                "Actual": pdata["hist_actual"],
                "Predicted": pdata["hist_pred"],
            }).set_index("Date")
            st.markdown("**Historical: Actual vs Predicted**")
            st.line_chart(hist_df)

            # Chart: forecast
            fut_df = pd.DataFrame({
                "Date": pdata["fut_dates"],
                "Forecast": pdata["fut_vals"],
            }).set_index("Date")
            st.markdown("**Forecast**")
            st.line_chart(fut_df)

            # Raw tables
            with st.expander("Raw data tables"):
                st.markdown("**Historical**")
                st.dataframe(
                    pd.DataFrame({"Date": pdata["hist_dates"], "Actual": pdata["hist_actual"], "Predicted": pdata["hist_pred"]}),
                    use_container_width=True
                )
                st.markdown("**Forecast**")
                st.dataframe(
                    pd.DataFrame({"Date": pdata["fut_dates"], "Forecast": pdata["fut_vals"]}),
                    use_container_width=True
                )


def page_notifications(cu):
    uid = cu["id"]
    st.title("🔔 Notifications")
    notifs = sorted(
        [n for n in _DB["notifs"].values() if n["user_id"] == uid],
        key=lambda x: x["created_at"], reverse=True
    )
    # mark all read
    for n in notifs:
        n["read"] = True

    if not notifs:
        st.info("No notifications yet.")
        return

    for n in notifs:
        icon = "✅" if n["ntype"] == "success" else "❌" if n["ntype"] == "error" else "ℹ️"
        st.markdown(f"{icon} {n['message']}  \n*{n['created_at'].strftime('%Y-%m-%d %H:%M')}*")
        st.markdown("---")


def page_profile(cu):
    st.title("👤 Profile")
    user = cu

    st.subheader("Update Info")
    with st.form("info_form"):
        name         = st.text_input("Full Name", value=user["name"])
        company      = st.text_input("Company", value=user.get("company", ""))
        role         = st.text_input("Role", value=user.get("role", "Analyst"))
        colors = ["#5d5fef","#dc2626","#2563eb","#16a34a","#db2777","#d97706","#7c3aed","#0891b2"]
        avatar_color = st.selectbox("Avatar Color", colors, index=colors.index(user["avatar_color"]) if user["avatar_color"] in colors else 0)
        save_info = st.form_submit_button("Save Info")
    if save_info:
        user["name"]         = name.strip() or user["name"]
        user["company"]      = company.strip()
        user["role"]         = role.strip() or "Analyst"
        user["avatar_color"] = avatar_color
        st.success("Profile updated.")
        st.rerun()

    st.subheader("Change Password")
    with st.form("pw_form"):
        current = st.text_input("Current Password", type="password")
        new_pw  = st.text_input("New Password", type="password")
        confirm = st.text_input("Confirm New Password", type="password")
        save_pw = st.form_submit_button("Update Password")
    if save_pw:
        if user["pw_hash"] != _hash(current):
            st.error("Current password is incorrect.")
        elif new_pw != confirm:
            st.error("New passwords do not match.")
        elif len(new_pw) < 6:
            st.error("Password must be at least 6 characters.")
        else:
            user["pw_hash"] = _hash(new_pw)
            st.success("Password updated successfully.")


def page_about():
    st.title("ℹ️ About SPREX NOVA")
    st.markdown("""
**SPREX NOVA** is a demand forecasting platform for spare parts and inventory management.

### Features
- Upload CSV / XLSX datasets and get AI-powered demand forecasts
- Supports **Random Forest** and **Linear Regression** models (auto-selected based on data size)
- Status labels: High Demand Expected, Overstock Risk, Risk of Stockout, Purchase Recommended, Demand Stable
- Per-part historical vs predicted charts and future forecast charts
- Notification centre, profile management, password change

### ML Engine
- Uses `scikit-learn` (RandomForestRegressor for ≥8 records, LinearRegression otherwise)
- Falls back to simple linear extrapolation if scikit-learn is unavailable
- MAE reported per part

### Tech Stack
- **Streamlit** frontend
- **pandas / numpy** data processing
- **scikit-learn** ML models
- In-memory session-based database
""")


# ═════════════════════════════════════════════════════════════════════════════
#  Router
# ═════════════════════════════════════════════════════════════════════════════
def main():
    # initialise page
    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    cu = get_current_user()
    page = st.session_state.get("page", "login")

    # redirect unauthenticated users
    if cu is None and page not in ("login", "signup", "forgot_password"):
        st.session_state["page"] = "login"
        page = "login"

    # redirect authenticated users away from auth pages
    if cu and page in ("login", "signup", "forgot_password"):
        st.session_state["page"] = "dashboard"
        page = "dashboard"

    # render sidebar for authenticated pages
    if cu:
        sidebar_nav(cu)

    # dispatch
    if page == "login":
        page_login()
    elif page == "signup":
        page_signup()
    elif page == "forgot_password":
        page_forgot_password()
    elif page == "dashboard":
        page_dashboard(cu)
    elif page == "upload":
        page_upload(cu)
    elif page == "results_list":
        page_results_list(cu)
    elif page == "results":
        page_results(cu)
    elif page == "notifications":
        page_notifications(cu)
    elif page == "profile":
        page_profile(cu)
    elif page == "about":
        page_about()
    else:
        st.session_state["page"] = "dashboard"
        st.rerun()


if __name__ == "__main__":
    main()