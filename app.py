"""
SPREX NOVA — Flask Backend
Vercel-compatible deployment
"""

import os, json, io, hashlib, secrets, re
from datetime import datetime, timedelta
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, g
)
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
#  App factory
# ═════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024   # 32 MB

# ─── In-memory "database" ────────────────────────────────────────────────────
# Structure:
#   USERS  : {email: {id, name, email, pw_hash, company, role, avatar_color, created_at}}
#   UPLOADS: {uid:   {id, user_id, filename, status, parts_count, uploaded_at, result_json}}
#   NOTIFS : {nid:   {id, user_id, ntype, message, read, created_at}}

_DB: dict = {"users": {}, "uploads": {}, "notifs": {}}

# ─── simple counter ids ──────────────────────────────────────────────────────
def _next_id(table: str) -> str:
    existing = [int(k) for k in _DB[table] if k.isdigit()]
    return str(max(existing, default=0) + 1)

# ─── password hashing ────────────────────────────────────────────────────────
def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# ═════════════════════════════════════════════════════════════════════════════
#  Auth helpers
# ═════════════════════════════════════════════════════════════════════════════
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please sign in first.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def get_current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    return next((u for u in _DB["users"].values() if u["id"] == uid), None)


def unread_notif_count(user_id: str) -> int:
    return sum(
        1 for n in _DB["notifs"].values()
        if n["user_id"] == user_id and not n["read"]
    )


@app.before_request
def _inject_globals():
    g.cu = get_current_user()
    g.nc = unread_notif_count(g.cu["id"]) if g.cu else 0


# ─── make cu / nc available in all templates ─────────────────────────────────
@app.context_processor
def _ctx():
    return {"cu": g.cu, "nc": g.nc}


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
#  ML Forecast engine
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
#  Routes — Auth
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    if g.cu:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))




@app.route("/login", methods=["GET", "POST"])
def login():
    if g.cu:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        pw    = request.form.get("password", "")
        user  = _DB["users"].get(email)
        if user and user["pw_hash"] == _hash(pw):
            session.clear()
            session["user_id"] = user["id"]
            # mark all notifs read on login
            flash(f"Welcome back, {user['name'].split()[0]}!", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.", "error")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if g.cu:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        name    = request.form.get("name", "").strip()
        email   = request.form.get("email", "").strip().lower()
        company = request.form.get("company", "").strip()
        pw      = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        if not name or not email or not pw:
            flash("Please fill all required fields.", "error")
        elif pw != confirm:
            flash("Passwords do not match.", "error")
        elif len(pw) < 6:
            flash("Password must be at least 6 characters.", "error")
        elif email in _DB["users"]:
            flash("An account with that email already exists.", "error")
        else:
            uid = _next_id("users")
            colors = ["#5d5fef","#dc2626","#2563eb","#16a34a","#db2777","#d97706","#7c3aed","#0891b2"]
            _DB["users"][email] = {
                "id": uid, "name": name, "email": email,
                "pw_hash": _hash(pw), "company": company, "role": "Analyst",
                "avatar_color": colors[int(uid) % len(colors)],
                "created_at": datetime.utcnow()
            }
            session.clear()
            session["user_id"] = uid
            _add_notif(uid, "success", "Welcome to SPREX NOVA! Upload your first dataset to get started.")
            flash("Account created! Welcome to SPREX NOVA.", "success")
            return redirect(url_for("dashboard"))
    return render_template("signup.html")


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email   = request.form.get("email", "").strip().lower()
        new_pw  = request.form.get("new_password", "")
        confirm = request.form.get("confirm", "")
        user    = _DB["users"].get(email)
        if not user:
            flash("No account found with that email.", "error")
        elif new_pw != confirm:
            flash("Passwords do not match.", "error")
        elif len(new_pw) < 6:
            flash("Password must be at least 6 characters.", "error")
        else:
            user["pw_hash"] = _hash(new_pw)
            flash("Password updated! Please sign in.", "success")
            return redirect(url_for("login"))
    return render_template("forgot_password.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You've been signed out.", "info")
    return redirect(url_for("login"))


# ═════════════════════════════════════════════════════════════════════════════
#  Routes — App
# ═════════════════════════════════════════════════════════════════════════════
@app.route("/dashboard")
@login_required
def dashboard():
    uid = g.cu["id"]
    user_uploads = [u for u in _DB["uploads"].values() if u["user_id"] == uid]
    user_uploads.sort(key=lambda x: x["uploaded_at"], reverse=True)
    total  = len(user_uploads)
    succ   = sum(1 for u in user_uploads if u["status"] == "success")
    total_parts = sum(u["parts_count"] for u in user_uploads if u["status"] == "success")
    recent = user_uploads[:5]
    return render_template("dashboard.html",
        total=total, succ=succ, total_parts=total_parts, recent=recent)


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    uid = g.cu["id"]
    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename:
            flash("No file selected.", "error")
            return redirect(url_for("upload"))

        filename = f.filename
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ("csv", "xlsx", "xls"):
            flash("Only CSV, XLSX, and XLS files are supported.", "error")
            return redirect(url_for("upload"))

        steps = int(request.form.get("steps", 6))
        try:
            raw = f.read()
            if ext == "csv":
                df = pd.read_csv(io.BytesIO(raw))
            else:
                df = pd.read_excel(io.BytesIO(raw), engine="openpyxl" if HAS_OPENPYXL else None)

            if df.empty or len(df.columns) < 2:
                raise ValueError("Dataset must have at least 2 columns.")

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
            flash(f"Forecast complete! {parts_count} part(s) analysed.", "success")
            return redirect(url_for("results", uid=rec_id))
        else:
            flash(f"Forecast failed: {result.get('error')}", "error")

    user_uploads = sorted(
        [u for u in _DB["uploads"].values() if u["user_id"] == uid],
        key=lambda x: x["uploaded_at"], reverse=True
    )
    return render_template("upload.html", uploads=user_uploads)


@app.route("/results")
@login_required
def results_list():
    uid = g.cu["id"]
    recs = sorted(
        [u for u in _DB["uploads"].values()
         if u["user_id"] == uid and u["status"] == "success"],
        key=lambda x: x["uploaded_at"], reverse=True
    )
    return render_template("results_list.html", recs=recs)


@app.route("/results/<uid>")
@login_required
def results(uid):
    rec = _DB["uploads"].get(uid)
    if not rec or rec["user_id"] != g.cu["id"]:
        flash("Result not found.", "error")
        return redirect(url_for("results_list"))
    data = json.loads(rec["result_json"])
    return render_template("results.html", rec=rec, data=data)


@app.route("/notifications")
@login_required
def notifications():
    uid = g.cu["id"]
    notifs = sorted(
        [n for n in _DB["notifs"].values() if n["user_id"] == uid],
        key=lambda x: x["created_at"], reverse=True
    )
    # mark all read
    for n in notifs:
        n["read"] = True
    return render_template("notifications.html", notifs=notifs)


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user = g.cu
    if request.method == "POST":
        action = request.form.get("action")
        if action == "info":
            user["name"]         = request.form.get("name", user["name"]).strip() or user["name"]
            user["company"]      = request.form.get("company", "").strip()
            user["role"]         = request.form.get("role", "Analyst").strip()
            user["avatar_color"] = request.form.get("avatar_color", user["avatar_color"])
            flash("Profile updated.", "success")
        elif action == "password":
            current = request.form.get("current", "")
            new_pw  = request.form.get("new", "")
            confirm = request.form.get("confirm", "")
            if user["pw_hash"] != _hash(current):
                flash("Current password is incorrect.", "error")
            elif new_pw != confirm:
                flash("New passwords do not match.", "error")
            elif len(new_pw) < 6:
                flash("Password must be at least 6 characters.", "error")
            else:
                user["pw_hash"] = _hash(new_pw)
                flash("Password updated successfully.", "success")
        return redirect(url_for("profile"))
    return render_template("profile.html")


@app.route("/about")
@login_required
def about():
    return render_template("about.html")


# ─── API endpoint for notification badge ─────────────────────────────────────
@app.route("/api/nc")
def api_nc():
    if not g.cu:
        return jsonify({"count": 0})
    return jsonify({"count": unread_notif_count(g.cu["id"])})


# ═════════════════════════════════════════════════════════════════════════════
#  Vercel entry-point
# ═════════════════════════════════════════════════════════════════════════════
# Vercel expects a WSGI callable named `app` at module level — it's already there.

if __name__ == "__main__":
    app.run(debug=True, port=5000)

app = app