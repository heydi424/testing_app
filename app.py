# app.py
# Non-Technical Legal Aid Analytics — Autopilot cleaning, analysis, maps (ZIP fallback), join, and Q&A
# Runtime: python-3.11

import io, re, math
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import folium
from streamlit.components.v1 import html as st_html

# Optional helpers (auto map via ZIP, optional SQL Q&A)
try:
    import pgeocode  # offline ZIP -> lat/lon
    HAS_PGEO = True
except Exception:
    HAS_PGEO = False

try:
    import duckdb
    HAS_DUCKDB = True
except Exception:
    HAS_DUCKDB = False

st.set_page_config(page_title="Legal Aid Analytics (Autopilot)", page_icon="⚖️", layout="wide")

ZIP_RE   = re.compile(r"^(\d{5})(?:-\d{4})?$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_DIGITS = re.compile(r"\D+")

# -------------------------
# Load data
# -------------------------
@st.cache_data(show_spinner=False)
def load_df(file) -> pd.DataFrame:
    try:
        if file.name.lower().endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)
    except Exception:
        if file.name.lower().endswith(".csv"):
            return pd.read_csv(io.BytesIO(file.getvalue()))
        return pd.read_excel(io.BytesIO(file.getvalue()))

# -------------------------
# Column profiling (plain-English)
# -------------------------
def is_phone_like(x: str) -> bool:
    d = PHONE_DIGITS.sub("", str(x))
    return (len(d) == 10) or (len(d) == 11 and d.startswith("1"))

def profile_column(s: pd.Series) -> Dict[str, object]:
    non_null = s.notna().sum()
    total = len(s)
    dtype = str(s.dtype)
    example = s.dropna().iloc[0] if non_null else None

    date_rate   = pd.to_datetime(s, errors="coerce").notna().mean()
    numeric_try = pd.to_numeric(pd.Series(s).astype(str).str.replace(",", ""), errors="coerce")
    numeric_rate = numeric_try.notna().mean()
    zip_rate    = s.astype(str).str.strip().str.extract(ZIP_RE)[0].notna().mean()
    email_rate  = s.astype(str).str.strip().str.lower().str.match(EMAIL_RE).fillna(False).mean()
    phone_rate  = (s.dropna().astype(str).map(is_phone_like).mean() if non_null else 0.0)
    lat_rate    = pd.to_numeric(s, errors="coerce").between(-90, 90).mean()
    lon_rate    = pd.to_numeric(s, errors="coerce").between(-180, 180).mean()

    uniq = s.dropna().nunique()
    cat_score = 1.0 - min(1.0, (uniq / max(1, non_null))) if non_null else 0.0

    tags = []
    if date_rate >= 0.5: tags.append("date")
    if numeric_rate >= 0.9: tags.append("numeric")
    if zip_rate >= 0.6: tags.append("zip")
    if email_rate >= 0.6: tags.append("email")
    if phone_rate >= 0.6: tags.append("phone")
    if lat_rate >= 0.9: tags.append("latitude")
    if lon_rate >= 0.9: tags.append("longitude")
    if "numeric" not in tags and "date" not in tags and cat_score >= 0.6 and uniq <= 60:
        tags.append("category")
    if not tags and dtype == "object" and uniq > 60:
        tags.append("text")

    return {
        "dtype": dtype,
        "missing_%": round((1 - (non_null / total)) * 100, 1) if total else 0.0,
        "unique_vals": int(uniq),
        "tags": ", ".join(tags),
        "example": example,
        "date_rate": float(date_rate),
        "numeric_rate": float(numeric_rate),
    }

def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        p = profile_column(df[c])
        rows.append({"column": c, **p})
    return pd.DataFrame(rows).sort_values(["tags", "column"])

# -------------------------
# Cleaning (automatic & safe)
# -------------------------
def clean_df(df: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Trim strings
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]):
            out[c] = out[c].astype(str).str.strip()

    # Dates
    date_cols = prof.loc[prof["tags"].str.contains("date", na=False), "column"].tolist()
    for c in date_cols:
        out[c] = pd.to_datetime(out[c], errors="coerce")

    # Emails
    for c in prof.loc[prof["tags"].str.contains("email", na=False), "column"]:
        out[c] = out[c].str.lower().where(out[c].str.match(EMAIL_RE, na=False), np.nan)

    # Phones
    for c in prof.loc[prof["tags"].str.contains("phone", na=False), "column"]:
        def fmt(x):
            if pd.isna(x): return np.nan
            d = PHONE_DIGITS.sub("", str(x))
            if len(d) == 10: return f"({d[:3]}) {d[3:6]}-{d[6:]}"
            if len(d) == 11 and d.startswith("1"):
                d = d[1:]; return f"+1 ({d[:3]}) {d[3:6]}-{d[6:]}"
            return np.nan
        out[c] = out[c].apply(fmt)

    # ZIP
    for c in prof.loc[prof["tags"].str.contains("zip", na=False), "column"]:
        out[c] = out[c].astype(str).str.strip().str.extract(ZIP_RE)[0]

    # Lat/Lon bounds
    for c in prof.loc[prof["tags"].str.contains("latitude", na=False), "column"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").where(lambda x: x.between(-90, 90))
    for c in prof.loc[prof["tags"].str.contains("longitude", na=False), "column"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").where(lambda x: x.between(-180, 180))

    # Mostly-numeric coercion
    mostly_numeric = prof[prof["numeric_rate"] >= 0.9]["column"].tolist()
    for c in mostly_numeric:
        out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", ""), errors="coerce")

    # Drop duplicates
    out = out.drop_duplicates()
    return out

# -------------------------
# Helper: choose “best” columns automatically (Autopilot)
# -------------------------
def choose_best(prof: pd.DataFrame, clean: pd.DataFrame):
    best_date = None
    cand = prof[prof["tags"].str.contains("date", na=False)]
    if not cand.empty:
        best_date = cand.sort_values("date_rate", ascending=False)["column"].iloc[0]

    cat_cols = prof[prof["tags"].str.contains("category", na=False)]["column"].tolist()
    cat_cols = [c for c in cat_cols if clean[c].nunique(dropna=True) >= 2][:6]
    best_cat = cat_cols[0] if cat_cols else None

    num_cols = [c for c in clean.columns if pd.api.types.is_numeric_dtype(clean[c]) and clean[c].notna().any()]
    best_num = num_cols[0] if num_cols else None

    lat = (prof[prof["tags"].str.contains("latitude", na=False)]["column"].tolist() or [None])[0]
    lon = (prof[prof["tags"].str.contains("longitude", na=False)]["column"].tolist() or [None])[0]
    zipc = (prof[prof["tags"].str.contains("zip", na=False)]["column"].tolist() or [None])[0]

    return dict(date=best_date, cat=best_cat, num=best_num, cat_list=cat_cols, num_list=num_cols, lat=lat, lon=lon, zip=zipc)

# -------------------------
# ZIP → coordinates (for map fallback)
# -------------------------
@st.cache_data(show_spinner=False)
def zip_to_latlon(zip_list: List[str]) -> pd.DataFrame:
    if not HAS_PGEO or not zip_list:
        return pd.DataFrame(columns=["postal_code","latitude","longitude"])
    nomi = pgeocode.Nominatim("us")
    res = nomi.query_postal_code(zip_list)
    # normalize columns
    res = res.rename(columns={"postal_code": "zip", "latitude": "lat", "longitude": "lon"})
    return res[["zip","lat","lon"]]

# -------------------------
# UI — super simple wizard
# -------------------------
st.sidebar.title("⚖️ Legal Aid Analytics — Autopilot")
st.sidebar.caption("Upload → Clean → (optional) Join → Filter → Visualize → Ask → Export")

up = st.sidebar.file_uploader("Upload your CSV/Excel", type=["csv","xlsx"], accept_multiple_files=False)
up_ext = st.sidebar.file_uploader("Upload external CSV/Excel (optional)", type=["csv","xlsx"], accept_multiple_files=False)

st.title("One-Click Data Analysis (for non-technical users)")

if not up:
    st.info("👆 Upload a file to begin. I’ll take it from there.")
    st.stop()

raw = load_df(up)
st.subheader("1) I found these columns in your file")
st.caption("No jargon—these are the actual headers plus friendly info.")
prof = profile_dataframe(raw)
st.dataframe(
    prof[["column","dtype","missing_%","unique_vals","tags","example"]],
    use_container_width=True, hide_index=True
)

st.subheader("2) I cleaned the data for you")
clean = clean_df(raw, prof)
st.success("Done: trimmed text, parsed dates, standardized email/phone/ZIP, fixed lat/lon, converted numbers, removed duplicates.")
with st.expander("See a quick preview"):
    st.dataframe(clean.head(50), use_container_width=True)

# Optional external join
work = clean.copy()
if up_ext:
    ext = load_df(up_ext)
    st.subheader("3) (Optional) Add extra data")
    left_key = st.selectbox("Pick a column in YOUR data to join on", options=list(work.columns))
    right_key = st.selectbox("Pick the matching column in the EXTERNAL data", options=list(ext.columns))
    if st.button("Join the files", type="primary"):
        work = work.merge(ext.rename(columns={right_key: left_key}), on=left_key, how="left")
        st.success(f"Joined on “{left_key}”.")
        with st.expander("Preview the joined data"):
            st.dataframe(work.head(50), use_container_width=True)

# Autopilot picks good defaults
choice = choose_best(prof, work)

st.subheader("4) Filters (optional)")
# Date range
filtered = work.copy()
if choice["date"] and filtered[choice["date"]].notna().any():
    dmin = pd.to_datetime(filtered[choice["date"]], errors="coerce").min()
    dmax = pd.to_datetime(filtered[choice["date"]], errors="coerce").max()
    if pd.notna(dmin) and pd.notna(dmax):
        start, end = st.date_input("Date range", value=(dmin.date(), dmax.date()))
        filtered = filtered[
            (pd.to_datetime(filtered[choice["date"]], errors="coerce") >= pd.to_datetime(start)) &
            (pd.to_datetime(filtered[choice["date"]], errors="coerce") <= pd.to_datetime(end))
        ]
else:
    st.caption("No clear date column detected—skipping date filter.")

# A few easy category filters (chips)
cat_cols = choice["cat_list"]
if cat_cols:
    with st.expander("Category filters"):
        for c in cat_cols[:4]:
            vals = sorted([str(x) for x in filtered[c].dropna().unique()][:300])
            sel = st.multiselect(f"{c}", ["(All)"] + vals, default=["(All)"])
            if sel and "(All)" not in sel:
                filtered = filtered[filtered[c].astype(str).isin(set(sel))]

# One or two numeric sliders
nums = choice["num_list"]
if nums:
    with st.expander("Number filters"):
        for c in nums[:2]:
            arr = pd.to_numeric(filtered[c], errors="coerce")
            if arr.notna().any():
                lo, hi = float(arr.min()), float(arr.max())
                if math.isfinite(lo) and math.isfinite(hi) and lo < hi:
                    a, b = st.slider(f"{c} range", min_value=lo, max_value=hi, value=(lo, hi))
                    filtered = filtered[(arr >= a) & (arr <= b)]

st.success(f"Filters applied — **{len(filtered)}** rows in view.")

# -------------------------
# 5) Friendly dashboard (Autopilot)
# -------------------------
st.subheader("5) Dashboard")

k1, k2, k3 = st.columns(3)
with k1: st.metric("Rows (cleaned)", len(work))
with k2: st.metric("Rows in view", len(filtered))
with k3: st.metric("Columns", filtered.shape[1])

with st.expander("See the data we’re charting"):
    st.dataframe(filtered.head(50), use_container_width=True)

# Time series
if choice["date"] and filtered[choice["date"]].notna().any():
    ts = filtered[[choice["date"]]].dropna().copy()
    ts["month"] = pd.to_datetime(ts[choice["date"]]).dt.to_period("M").dt.to_timestamp()
    ts = ts.groupby("month").size().reset_index(name="count")
    st.plotly_chart(px.line(ts, x="month", y="count", title="Activity over time"), use_container_width=True)

# Top category
if choice["cat"] and filtered[choice["cat"]].notna().any():
    topc = filtered[choice["cat"]].value_counts().head(12).reset_index()
    topc.columns = [choice["cat"], "count"]
    st.plotly_chart(px.bar(topc, x=choice["cat"], y="count", title=f"Top values — {choice['cat']}"), use_container_width=True)

# Numeric distribution
if choice["num"] and pd.api.types.is_numeric_dtype(filtered[choice["num"]]) and filtered[choice["num"]].notna().any():
    st.plotly_chart(px.histogram(filtered, x=choice["num"], nbins=30, title=f"Distribution — {choice['num']}"), use_container_width=True)

# Correlation (if enough numerics)
num_cols_all = [c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c]) and filtered[c].notna().any()]
if len(num_cols_all) >= 2:
    corr = filtered[num_cols_all].corr(numeric_only=True)
    st.plotly_chart(px.imshow(corr, text_auto=True, title="Numbers that move together"), use_container_width=True)

# Map — uses lat/lon if present; otherwise uses ZIP centroid fallback
st.subheader("Map")
lat_c, lon_c, zip_c = choice["lat"], choice["lon"], choice["zip"]
if lat_c and lon_c and filtered[[lat_c, lon_c]].notna().any().any():
    pts = filtered.dropna(subset=[lat_c, lon_c])
    if not pts.empty:
        m = folium.Map(location=[pts[lat_c].mean(), pts[lon_c].mean()], zoom_start=9)
        for _, r in pts.iterrows():
            folium.CircleMarker([r[lat_c], r[lon_c]], radius=4).add_to(m)
        st_html(m._repr_html_(), height=520)
    else:
        st.info("No rows left with both latitude and longitude after filters.")
elif zip_c and filtered[zip_c].notna().any() and HAS_PGEO:
    z = filtered[zip_c].astype(str).str.extract(r"(\d{5})")[0].dropna()
    if not z.empty:
        counts = z.value_counts().rename_axis("zip").reset_index(name="count")
        geo = zip_to_latlon(counts["zip"].tolist())
        zz = counts.merge(geo, how="left", on="zip").dropna(subset=["lat","lon"])
        if not zz.empty:
            m = folium.Map(location=[zz["lat"].mean(), zz["lon"].mean()], zoom_start=7)
            for _, r in zz.iterrows():
                folium.Circle(
                    location=[r["lat"], r["lon"]],
                    radius=200 + 50 * float(r["count"]),
                    popup=f"ZIP {r['zip']}: {int(r['count'])}"
                ).add_to(m)
            st_html(m._repr_html_(), height=520)
        else:
            st.info("I couldn’t look up ZIP locations for mapping.")
    else:
        st.info("No 5-digit ZIPs found for mapping.")
else:
    st.caption("No location fields yet — map is optional. Add a ZIP column or lat/lon for a map.")

# -------------------------
# 6) Ask a question (simple buttons + natural language)
# -------------------------
st.subheader("6) Ask a question")
st.caption("Click a quick question, or type your own in plain English. (No coding.)")

# Quick chips based on available columns
chips = []
if choice["cat"]:
    chips += [f"count by {choice['cat']}", f"top 10 {choice['cat']}"]
if choice["num"] and choice["cat"]:
    chips += [f"average {choice['num']} by {choice['cat']}"]
if zip_c:
    chips += ["top 10 " + zip_c]

cols = st.columns(min(4, max(1, len(chips))))
query = st.session_state.get("qa_query", "")
for i, text in enumerate(chips[:8]):
    if cols[i % len(cols)].button(text):
        query = text
        st.session_state["qa_query"] = text

query = st.text_input("Your question", value=query, placeholder="e.g., count by Department")
go = st.button("Run")

def answer_nl(q: str, df: pd.DataFrame):
    q = q.strip().lower()
    if not q: return None, "Type or click a question above."
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
        if x in df.columns and y in df.columns and pd.api.types.is_numeric_dtype(df[x]):
            func = "mean" if agg in ("average","mean") else "median"
            out = df.groupby(y)[x].agg(func).reset_index().sort_values(x, ascending=False)
            return out, f"{agg.title()} {x} by {y}"
    # show rows where col=value
    m = re.match(r"(rows|show)\s+where\s+([a-z0-9_]+)\s*={1,2}\s*([a-z0-9_\-@.]+)", q)
    if m:
        col, val = m.group(2), m.group(3)
        if col in df.columns:
            out = df[df[col].astype(str).str.lower() == val.lower()]
            return out.head(100), f"Rows where {col} == {val} (first 100)"
    # how many rows
    if q in ("how many rows","row count","count rows"):
        return pd.DataFrame({"rows":[len(df)]}), "Row count"
    return None, "Try: 'count by <column>', 'top 10 <column>', 'average <number> by <category>'."

if go:
    res, msg = answer_nl(query, filtered)
    st.caption(msg)
    if isinstance(res, pd.DataFrame):
        st.dataframe(res, use_container_width=True)

# -------------------------
# 7) One-click report & export
# -------------------------
st.subheader("7) Download report & data")

def build_report(df: pd.DataFrame, choice) -> str:
    lines = []
    lines.append("# Automated Analysis Report")
    lines.append(f"**Rows in view:** {len(df)}")
    if choice["date"] and choice["date"] in df.columns and df[choice["date"]].notna().any():
        dmin = pd.to_datetime(df[choice["date"]]).min().date()
        dmax = pd.to_datetime(df[choice["date"]]).max().date()
        lines.append(f"**Date span ({choice['date']}):** {dmin} → {dmax}")
    # top of first few low-cardinality columns
    small_cats = [c for c in df.columns if (df[c].dtype == "object" or pd.api.types.is_categorical_dtype(df[c])) and df[c].nunique(dropna=True) <= 50]
    for c in small_cats[:3]:
        top = df[c].value_counts().head(5)
        if not top.empty:
            lines.append(f"\n**Top `{c}`:**")
            for k, v in top.items():
                lines.append(f"- {k}: {v}")
    # numeric summary
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if nums:
        desc = df[nums].describe().T.round(2)
        lines.append("\n**Numeric summary:**\n")
        lines.append(desc.to_markdown())
    return "\n".join(lines)

report_md = build_report(filtered, choice)
st.download_button("⬇️ Download analysis report (.md)", report_md.encode(), file_name="analysis_report.md", mime="text/markdown")

with io.StringIO() as buffer:
    filtered.to_csv(buffer, index=False)
    st.download_button("Download filtered CSV", data=buffer.getvalue().encode(), file_name="filtered_data.csv", mime="text/csv")
