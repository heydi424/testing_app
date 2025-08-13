# app.py
# Zero-Assumption Legal Aid EDA — auto-detect, auto-clean, analyze, join, Q&A
# Runtime: python-3.11

import io, re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import folium
from streamlit.components.v1 import html as st_html

# Optional: SQL for ad-hoc questions
try:
    import duckdb
    HAS_DUCKDB = True
except Exception:
    HAS_DUCKDB = False

st.set_page_config(page_title="Legal Aid Analytics (Zero-Assumption)", page_icon="⚖️", layout="wide")

ZIP_RE   = re.compile(r"^(\d{5})(?:-\d{4})?$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_DIGITS = re.compile(r"\D+")

# -------------------------
# Loaders
# -------------------------
@st.cache_data(show_spinner=False)
def load_df(file) -> pd.DataFrame:
    try:
        if file.name.lower().endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)
    except Exception:
        # bytes fallback
        if file.name.lower().endswith(".csv"):
            return pd.read_csv(io.BytesIO(file.getvalue()))
        return pd.read_excel(io.BytesIO(file.getvalue()))

# -------------------------
# Detection heuristics (no header assumptions)
# -------------------------
def frac_dates(s: pd.Series) -> float:
    return pd.to_datetime(s, errors="coerce").notna().mean()

def frac_zip(s: pd.Series) -> float:
    return s.astype(str).str.strip().str.extract(ZIP_RE)[0].notna().mean()

def frac_email(s: pd.Series) -> float:
    return s.astype(str).str.strip().str.lower().str.match(EMAIL_RE).fillna(False).mean()

def frac_phone_like(s: pd.Series) -> float:
    def ok(v):
        d = PHONE_DIGITS.sub("", str(v))
        return (len(d) == 10) or (len(d) == 11 and d.startswith("1"))
    x = s.dropna().astype(str)
    if x.empty: return 0.0
    return x.map(ok).mean()

def frac_lat(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return s.between(-90, 90).mean()

def frac_lon(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return s.between(-180, 180).mean()

def cat_score(s: pd.Series) -> float:
    n = s.dropna().shape[0]
    if n == 0: return 0.0
    uniq = s.dropna().nunique()
    # higher score => more categorical
    return 1.0 - min(1.0, uniq / max(1, n))

def detect_schema(df: pd.DataFrame) -> Dict[str, any]:
    schema = {
        "date_cols": [],
        "best_date": None,
        "zip_col": None,
        "email_col": None,
        "phone_col": None,
        "lat_col": None,
        "lon_col": None,
        "numeric_cols": [],
        "categorical_cols": [],
        "text_cols": [],
    }
    if df.empty:
        return schema

    # Basic type buckets
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # Try to parse numerics from strings where obviously numeric
    for c in obj_cols:
        try:
            coerce_try = pd.to_numeric(df[c].str.replace(",", ""), errors="coerce")
            if coerce_try.notna().mean() > 0.9:
                num_cols.append(c)
        except Exception:
            pass

    # Dates (score by successful parse ratio)
    date_scores = {c: frac_dates(df[c]) for c in df.columns}
    date_candidates = [c for c, s in date_scores.items() if s >= 0.4]  # at least 40% parseable as dates
    best_date = max(date_candidates, key=lambda c: date_scores[c]) if date_candidates else None

    # ZIP / email / phone
    zip_candidates = [(c, frac_zip(df[c])) for c in df.columns]
    zip_candidates = sorted(zip_candidates, key=lambda x: x[1], reverse=True)
    zip_col = zip_candidates[0][0] if zip_candidates and zip_candidates[0][1] >= 0.5 else None

    email_candidates = [(c, frac_email(df[c])) for c in obj_cols]
    email_candidates = sorted(email_candidates, key=lambda x: x[1], reverse=True)
    email_col = email_candidates[0][0] if email_candidates and email_candidates[0][1] >= 0.5 else None

    phone_candidates = [(c, frac_phone_like(df[c])) for c in df.columns]
    phone_candidates = sorted(phone_candidates, key=lambda x: x[1], reverse=True)
    phone_col = phone_candidates[0][0] if phone_candidates and phone_candidates[0][1] >= 0.5 else None

    # Lat/Lon
    lat_candidates = [(c, frac_lat(df[c])) for c in df.columns]
    lat_candidates.sort(key=lambda x: x[1], reverse=True)
    lat_col = lat_candidates[0][0] if lat_candidates and lat_candidates[0][1] >= 0.8 else None

    lon_candidates = [(c, frac_lon(df[c])) for c in df.columns]
    lon_candidates.sort(key=lambda x: x[1], reverse=True)
    lon_col = lon_candidates[0][0] if lon_candidates and lon_candidates[0][1] >= 0.8 else None

    # Categorical candidates: low unique fraction; cap by unique count
    cat_cols = []
    for c in df.columns:
        score = cat_score(df[c].astype(str))
        if score >= 0.6 and df[c].nunique(dropna=True) <= 50:
            cat_cols.append(c)

    # Text-ish (fallback for free text; large cardinality)
    text_cols = [c for c in obj_cols if df[c].nunique(dropna=True) > 50]

    schema.update({
        "date_cols": [c for c in date_candidates],
        "best_date": best_date,
        "zip_col": zip_col,
        "email_col": email_col,
        "phone_col": phone_col,
        "lat_col": lat_col,
        "lon_col": lon_col,
        "numeric_cols": list(dict.fromkeys(num_cols)),
        "categorical_cols": cat_cols,
        "text_cols": text_cols,
    })
    return schema

# -------------------------
# Cleaning (only for detected columns)
# -------------------------
def clean_zip_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.extract(ZIP_RE)[0]

def clean_phone_series(s: pd.Series) -> pd.Series:
    def fmt(x):
        if pd.isna(x): return np.nan
        d = PHONE_DIGITS.sub("", str(x))
        if len(d)==10: return f"({d[:3]}) {d[3:6]}-{d[6:]}"
        if len(d)==11 and d.startswith("1"): 
            d = d[1:]; return f"+1 ({d[:3]}) {d[3:6]}-{d[6:]}"
        return np.nan
    return s.apply(fmt)

def clean_df_by_schema(df: pd.DataFrame, s: Dict[str, any]) -> pd.DataFrame:
    out = df.copy()
    # standard trims for objects
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype(str).str.strip()

    # dates
    for c in s["date_cols"]:
        out[c] = pd.to_datetime(out[c], errors="coerce")

    # zip/email/phone
    if s["zip_col"]:
        out[s["zip_col"]] = clean_zip_series(out[s["zip_col"]])
    if s["email_col"]:
        out[s["email_col"]] = out[s["email_col"]].str.lower().where(
            out[s["email_col"]].str.match(EMAIL_RE, na=False), np.nan)
    if s["phone_col"]:
        out[s["phone_col"]] = clean_phone_series(out[s["phone_col"]])

    # lat/lon numeric bounds
    if s["lat_col"]:
        out[s["lat_col"]] = pd.to_numeric(out[s["lat_col"]], errors="coerce").where(lambda x: x.between(-90,90))
    if s["lon_col"]:
        out[s["lon_col"]] = pd.to_numeric(out[s["lon_col"]], errors="coerce").where(lambda x: x.between(-180,180))

    # drop perfect duplicate rows
    out = out.drop_duplicates()
    return out

# -------------------------
# Sidebar: uploads
# -------------------------
st.sidebar.title("⚖️ Legal Aid Analytics (Zero-Assumption)")
st.sidebar.caption("Upload → Auto-detect → Clean → (optional) Join → Analyze → Ask → Export")

up = st.sidebar.file_uploader("Internal CSV/Excel", type=["csv","xlsx"], accept_multiple_files=False)
up_ext = st.sidebar.file_uploader("External CSV/Excel (optional)", type=["csv","xlsx"], accept_multiple_files=False)

st.title("Zero-Assumption Data Science Workflow")

if not up:
    st.info("👆 Upload a CSV or Excel file to begin.")
    st.stop()

raw = load_df(up)
schema = detect_schema(raw)
clean = clean_df_by_schema(raw, schema)

# -------------------------
# External join (user picks keys; no assumptions)
# -------------------------
work = clean.copy()
if up_ext:
    ext = load_df(up_ext)
    st.subheader("External Data Join")
    c1, c2 = st.columns(2)
    with c1:
        left_key = st.selectbox("Internal join column", options=list(work.columns), index=0)
    with c2:
        right_key = st.selectbox("External join column", options=list(ext.columns), index=0)
    if st.button("Join external data", type="primary"):
        work = work.merge(ext.rename(columns={right_key: left_key}), on=left_key, how="left")
        st.success(f"Joined on `{left_key}`.")

# -------------------------
# Filters (created only from detected, non-empty columns)
# -------------------------
st.subheader("Filters")

# Date filter
date_field = None
if schema["best_date"] and work[schema["best_date"]].notna().any():
    date_field = schema["best_date"]
    min_d = pd.to_datetime(work[date_field], errors="coerce").min()
    max_d = pd.to_datetime(work[date_field], errors="coerce").max()
    if pd.notna(min_d) and pd.notna(max_d):
        start_date, end_date = st.date_input("Date range", value=(min_d.date(), max_d.date()))
    else:
        start_date, end_date = None, None
else:
    st.caption("No date column detected for time filtering.")
    start_date, end_date = None, None

# Categorical filters (pick up to 6 most categorical)
cat_cols = [c for c in schema["categorical_cols"] if work[c].notna().any()]
cat_cols = cat_cols[:6]
cat_filters = {}
if cat_cols:
    cols = st.columns(min(4, len(cat_cols)))
    for i, c in enumerate(cat_cols):
        with cols[i % len(cols)]:
            vals = sorted([str(x) for x in work[c].dropna().unique()][:400])
            sel = st.multiselect(c, ["(All)"] + vals, default=["(All)"])
            cat_filters[c] = sel

# Apply filters
f = work.copy()
if date_field and start_date and end_date:
    f = f[(pd.to_datetime(f[date_field], errors="coerce") >= pd.to_datetime(start_date)) &
          (pd.to_datetime(f[date_field], errors="coerce") <= pd.to_datetime(end_date))]
for c, sel in cat_filters.items():
    if sel and "(All)" not in sel:
        f = f[f[c].astype(str).isin(set(sel))]

# -------------------------
# Data quality snapshot
# -------------------------
st.subheader("Data Quality")
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Rows (cleaned)", len(clean))
with c2: st.metric("Rows (in view)", len(f))
with c3: st.metric("Detected date cols", len(schema["date_cols"]))
with c4: st.metric("Categorical cols", len(schema["categorical_cols"]))
with st.expander("Detected schema"):
    st.json(schema)

# -------------------------
# EDA visuals (generic)
# -------------------------
st.subheader("Overview & Trends")

# Time series (if date)
if date_field and f[date_field].notna().any():
    ts = f[[date_field]].dropna().copy()
    ts["month"] = pd.to_datetime(ts[date_field]).dt.to_period("M").dt.to_timestamp()
    ts = ts.groupby("month").size().reset_index(name="count")
    st.plotly_chart(px.line(ts, x="month", y="count", title=f"Records per month ({date_field})"), use_container_width=True)

# Top categories (first categorical col)
if schema["categorical_cols"]:
    cat = schema["categorical_cols"][0]
    if cat in f.columns and f[cat].notna().any():
        topc = f[cat].value_counts().head(10).reset_index()
        topc.columns = [cat, "count"]
        st.plotly_chart(px.bar(topc, x=cat, y="count", title=f"Top values — {cat}"), use_container_width=True)

# Numeric distributions (first 2)
num_cols = [c for c in schema["numeric_cols"] if pd.api.types.is_numeric_dtype(f[c]) and f[c].notna().any()]
if num_cols:
    for c in num_cols[:2]:
        st.plotly_chart(px.histogram(f, x=c, nbins=30, title=f"Distribution — {c}"), use_container_width=True)

# Correlation (if enough numerics)
if len(num_cols) >= 2:
    corr = f[num_cols].corr(numeric_only=True)
    st.plotly_chart(px.imshow(corr, text_auto=True, title="Numeric correlation"), use_container_width=True)

# -------------------------
# Maps (only if lat/lon detected)
# -------------------------
st.subsection = st.subheader  # alias
st.subheader("Map")
lat_c, lon_c = schema["lat_col"], schema["lon_col"]
if lat_c and lon_c and f[[lat_c, lon_c]].notna().any().any():
    pts = f.dropna(subset=[lat_c, lon_c])
    if not pts.empty:
        m = folium.Map(location=[pts[lat_c].mean(), pts[lon_c].mean()], zoom_start=9)
        for _, r in pts.iterrows():
            folium.CircleMarker([r[lat_c], r[lon_c]], radius=4).add_to(m)
        st_html(m._repr_html_(), height=520)
else:
    # ZIP hotspot table as a fallback
    zip_c = schema["zip_col"]
    if zip_c and f[zip_c].notna().any():
        ztab = f.groupby(zip_c).size().reset_index(name="count").sort_values("count", ascending=False)
        st.dataframe(ztab, use_container_width=True)
    else:
        st.info("No latitude/longitude or ZIP detected, so no geospatial view.")

# -------------------------
# Q&A (NL patterns or SQL)
# -------------------------
st.subheader("Ask a question")
q1, q2 = st.columns([3,1])
question = q1.text_input("Ask in plain language (e.g., 'count by <column>', 'top 10 <column>', 'average <num> by <cat>') or SQL.")
sql_mode = q2.toggle("SQL mode (DuckDB)", value=False)

def answer_nl(q: str, df: pd.DataFrame):
    q = q.strip().lower()
    if not q: return None, "Type a question."
    # top N of column
    m = re.match(r"top\s+(\d+)\s+([a-z0-9_]+)", q)
    if m:
        n, col = int(m.group(1)), m.group(2)
        if col in df.columns:
            out = df[col].value_counts().head(n).reset_index()
            out.columns = [col, "count"]; return out, f"Top {n} of {col}"
    # count by col
    m = re.match(r"(count|how many)\s+(by|per)\s+([a-z0-9_]+)", q)
    if m:
        col = m.group(3)
        if col in df.columns:
            out = df.groupby(col).size().reset_index(name="count").sort_values("count", ascending=False)
            return out, f"Count by {col}"
    # avg/median X by Y
    m = re.match(r"(average|mean|median)\s+([a-z0-9_]+)\s+(by|per)\s+([a-z0-9_]+)", q)
    if m:
        agg, x, y = m.group(1), m.group(2), m.group(4)
        if x in df.columns and y in df.columns:
            func = "mean" if agg in ("average","mean") else "median"
            out = df.groupby(y)[x].agg(func).reset_index().sort_values(x, ascending=False)
            return out, f"{agg.title()} {x} by {y}"
    # rows where col=value
    m = re.match(r"(rows|show)\s+where\s+([a-z0-9_]+)\s*={1,2}\s*([a-z0-9_\-@.]+)", q)
    if m:
        col, val = m.group(2), m.group(3)
        if col in df.columns:
            out = df[df[col].astype(str).str.lower() == val.lower()]
            return out.head(100), f"Rows where {col} == {val} (first 100)"
    # row count
    if q in ("how many rows","row count","count rows"):
        return pd.DataFrame({"rows":[len(df)]}), "Row count"
    return None, "Try: 'count by <column>', 'top 10 <column>', 'average <num> by <cat>', or use SQL mode."

if st.button("Run"):
    if sql_mode:
        if not HAS_DUCKDB:
            st.error("DuckDB not available. Add `duckdb` to requirements.txt or turn off SQL mode.")
        else:
            try:
                duckdb.sql("DROP VIEW IF EXISTS df")
                duckdb.register("df", f)
                res = duckdb.sql(question).df()
                st.dataframe(res, use_container_width=True)
            except Exception as e:
                st.error(f"SQL error: {e}")
    else:
        res, msg = answer_nl(question, f)
        st.caption(msg)
        if isinstance(res, pd.DataFrame):
            st.dataframe(res, use_container_width=True)

# -------------------------
# Report + Export
# -------------------------
st.subheader("Report & Export")

def build_report(df: pd.DataFrame) -> str:
    lines = []
    lines.append("# Automated EDA Report")
    lines.append(f"**Rows in view:** {len(df)}")
    # date span
    if date_field and df[date_field].notna().any():
        lines.append(f"**Date span:** {pd.to_datetime(df[date_field]).min().date()} → {pd.to_datetime(df[date_field]).max().date()}")
    # top categories
    for c in schema["categorical_cols"][:3]:
        if c in df.columns and df[c].notna().any():
            top = df[c].value_counts().head(5)
            lines.append(f"**Top of `{c}`:**")
            for k, v in top.items():
                lines.append(f"- {k}: {v}")
    # numeric summaries
    nums = [c for c in schema["numeric_cols"] if c in df.columns]
    if nums:
        desc = df[nums].describe().T.round(2)
        lines.append("\n**Numeric summary (describe):**\n")
        lines.append(desc.to_markdown())
    return "\n".join(lines)

report_md = build_report(f)
st.markdown(report_md)
st.download_button("⬇️ Download report (.md)", report_md.encode(), file_name="eda_report.md", mime="text/markdown")

with io.StringIO() as buffer:
    f.to_csv(buffer, index=False)
    csv_bytes = buffer.getvalue().encode()
st.download_button("Download filtered CSV", data=csv_bytes, file_name="filtered_data.csv", mime="text/csv")
