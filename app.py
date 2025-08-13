# app.py
# Streamlit Analytics App for a Nonprofit Legal Aid Organization — Schema Wizard Edition
# -------------------------------------------------------------------------------------
# Run locally (safe in a separate env): streamlit run app.py
# Features:
# - Upload any CSV/Excel (unknown headers OK)
# - Schema Wizard to map your file's columns → standard roles (department, opened_date, zip, etc.)
# - Automatic cleaning by role (dates, phone, email, ZIP, categoricals)
# - Optional external data join (pick external join column + internal role)
# - Interactive dashboards: trends, issues/events, outcomes, demographics
# - Maps: points (lat/long) and ZIP hotspots via centroids
# - Download filtered CSV

import io
import json
import re
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import folium
from streamlit.components.v1 import html as st_html

# Optional fuzzy matcher for better auto-suggestions
try:
    from rapidfuzz import process as fuzz_process
    HAS_FUZZ = True
except Exception:
    HAS_FUZZ = False

st.set_page_config(page_title="Legal Aid Analytics", page_icon="⚖️", layout="wide")

# -------------------------
# Constants & Cleaning Config
# -------------------------
DEPT_OPTIONS = [
    "All Departments",
    "Family",
    "Expungement",
    "Consumer Law",
    "Tenant Rights & Housing",
    "Administrative Law",
    "Outreach",
    "Marketing",
    "Language Access",
]

STANDARD_ROLES = [
    "client_id","department","legal_issue","opened_date","closed_date",
    "zip","city","county","state","language","event_type","event_date",
    "outcome","referral_source","income_pct_fpl","age","race_ethnicity",
    "gender","phone","email","address","latitude","longitude"
]

ROLE_SYNONYMS = {
    "client_id": ["client id","clientid","person_id","case_client_id","id"],
    "department": ["dept","unit","program"],
    "legal_issue": ["issue","case_type","matter","topic"],
    "opened_date": ["open date","intake_date","created","start_date"],
    "closed_date": ["close date","resolved","end_date"],
    "zip": ["zipcode","postal","postal_code","zip_code","zip5"],
    "city": ["municipality","town"],
    "county": ["parish","borough"],
    "state": ["province","state_code"],
    "language": ["primary_language","lang"],
    "event_type": ["outreach_type","event kind"],
    "event_date": ["outreach_date","eventdate"],
    "outcome": ["disposition","result"],
    "referral_source": ["how heard","referrer","source"],
    "income_pct_fpl": ["fpl","% fpl","income fpl"],
    "age": ["client_age","age_years"],
    "race_ethnicity": ["race","ethnicity","race/ethnicity"],
    "gender": ["sex","gender_identity"],
    "phone": ["phone_number","mobile","cell"],
    "email": ["email_address","e-mail"],
    "address": ["street","addr","address1"],
    "latitude": ["lat"],
    "longitude": ["lon","lng","long"],
}

DEFAULT_ROLE_CLEANING = {
    "opened_date": {"type": "date"},
    "closed_date": {"type": "date", "allow_null": True},
    "event_date": {"type": "date", "allow_null": True},
    "zip": {"type": "zip"},
    "phone": {"type": "phone", "allow_null": True},
    "email": {"type": "email", "allow_null": True},
    "income_pct_fpl": {"type": "numeric", "min": 0, "max": 600},
    "age": {"type": "numeric", "min": 0, "max": 120},
    "latitude": {"type": "numeric", "min": -90, "max": 90, "allow_null": True},
    "longitude": {"type": "numeric", "min": -180, "max": 180, "allow_null": True},
    "department": {"type": "category", "map": {
        "family": "Family", "housing": "Tenant Rights & Housing", "tenant": "Tenant Rights & Housing",
        "expunge": "Expungement", "consumer": "Consumer Law", "admin": "Administrative Law",
        "outreach": "Outreach", "marketing": "Marketing", "language": "Language Access"
    }},
    "language": {"type": "category", "standardize_case": True},
    "state": {"type": "category", "upper": True},
    "city": {"type": "category", "title": True},
    "county": {"type": "category", "title": True},
    "legal_issue": {"type": "text_trim"},
    "referral_source": {"type": "text_trim"},
    "outcome": {"type": "text_trim"},
    "race_ethnicity": {"type": "text_trim"},
    "gender": {"type": "text_trim"},
    "address": {"type": "text_trim"},
}

ZIP_RE = re.compile(r"^(\d{5})(?:-\d{4})?$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# -------------------------
# Loaders & Cleaners
# -------------------------
@st.cache_data(show_spinner=False)
def load_df(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

@st.cache_data(show_spinner=False)
def suggest_mapping(columns: List[str]):
    suggestions = {}
    norm_cols = {c: _norm(c) for c in columns}
    for role in STANDARD_ROLES:
        # exact
        for c, n in norm_cols.items():
            if n == _norm(role) or n.replace(" ", "_") == role:
                suggestions[role] = c; break
        if role in suggestions: 
            continue
        # synonym
        for syn in ROLE_SYNONYMS.get(role, []):
            for c, n in norm_cols.items():
                if n == _norm(syn):
                    suggestions[role] = c; break
            if role in suggestions: 
                break
        # optional fuzzy
        if not role in suggestions and HAS_FUZZ:
            best = fuzz_process.extractOne(role, list(norm_cols.values()))
            if best and best[1] >= 85:
                for c, n in norm_cols.items():
                    if n == best[0]:
                        suggestions[role] = c; break
    return suggestions

# Cleaning helpers
def to_date(s): return pd.to_datetime(s, errors="coerce")
def to_num(s, lo=None, hi=None):
    s = pd.to_numeric(s, errors="coerce")
    if lo is not None: s = s.where(s >= lo)
    if hi is not None: s = s.where(s <= hi)
    return s
def clean_zip(s): return s.astype(str).str.strip().str.extract(ZIP_RE)[0]
def clean_phone(s):
    def f(x):
        if pd.isna(x): return np.nan
        digits = re.sub(r"\D+", "", str(x))
        if len(digits)==10: return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"
        if len(digits)==11 and digits.startswith("1"):
            d = digits[1:]; return f"+1 ({d[0:3]}) {d[3:6]}-{d[6:10]}"
        return np.nan
    return s.apply(f)
def clean_email(s): return s.astype(str).str.strip().str.lower().where(lambda x: x.str.match(EMAIL_RE), np.nan)
def case_map(s, mapping=None, upper=False, title=False, standardize_case=False):
    if mapping:
        return s.apply(lambda x: mapping.get(str(x).strip().lower(), str(x).strip()) if pd.notna(x) else np.nan)
    if upper:  return s.apply(lambda x: str(x).strip().upper() if pd.notna(x) else np.nan)
    if title:  return s.apply(lambda x: str(x).strip().title() if pd.notna(x) else np.nan)
    if standardize_case: return s.apply(lambda x: str(x).strip().title() if pd.notna(x) else np.nan)
    return s
def trim(s): return s.apply(lambda x: str(x).strip() if pd.notna(x) else x)

OPS = {
    "date": lambda s,c: to_date(s),
    "numeric": lambda s,c: to_num(s, c.get("min"), c.get("max")),
    "zip": lambda s,c: clean_zip(s),
    "phone": lambda s,c: clean_phone(s),
    "email": lambda s,c: clean_email(s),
    "category": lambda s,c: case_map(s, c.get("map"), c.get("upper", False), c.get("title", False), c.get("standardize_case", False)),
    "text_trim": lambda s,c: trim(s),
}

@st.cache_data(show_spinner=False)
def apply_mapping_and_clean(df: pd.DataFrame, role_map: Dict[str,str], cleaning=DEFAULT_ROLE_CLEANING):
    df = df.copy()
    unified = pd.DataFrame(index=df.index)
    for role in STANDARD_ROLES:
        src = role_map.get(role)
        unified[role] = df[src] if (src is not None and src in df.columns) else np.nan

    issues = {}
    for role, cfg in cleaning.items():
        if role in unified.columns:
            kind = cfg.get("type")
            if kind in OPS:
                before = unified[role].isna().sum()
                unified[role] = OPS[kind](unified[role], cfg)
                after = unified[role].isna().sum()
                if after > before:
                    issues.setdefault("values_nullified", []).append(f"{role}: {after-before}")

    subset = [c for c in ["client_id","department","opened_date","legal_issue","zip"] if c in unified.columns]
    if subset:
        before = len(unified)
        unified = unified.drop_duplicates(subset=subset, keep="first")
        removed = before - len(unified)
        if removed: issues.setdefault("duplicates_removed", []).append(str(removed))

    issues["unmapped_roles"] = [r for r in STANDARD_ROLES if not role_map.get(r)]
    return unified, issues

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("⚖️ Legal Aid Analytics")
st.sidebar.caption("Upload → Map → Clean → Join → Visualize → Export")

with st.sidebar.expander("1) Upload Internal Data", expanded=True):
    up = st.file_uploader("Internal CSV/Excel (cases/outreach)", type=["csv","xlsx"], accept_multiple_files=False)

with st.sidebar.expander("2) Upload External Data (optional)"):
    up_ext = st.file_uploader("External CSV/Excel (e.g., Census by ZIP/County/State)", type=["csv","xlsx"], accept_multiple_files=False)

with st.sidebar.expander("Filters", expanded=True):
    dept = st.selectbox("Department", options=DEPT_OPTIONS, index=0)
    date_field = st.selectbox("Date field", options=["opened_date","event_date","closed_date"]) 
    start_date = st.date_input("Start", value=pd.to_datetime("2023-01-01").date())
    end_date = st.date_input("End", value=pd.to_datetime("today").date())

# -------------------------
# Main: Schema Wizard & Data Flow
# -------------------------
st.title("Legal Aid Analytics — Streamlit App")
st.write("Upload a file, map its headers to standard roles with the **Schema Wizard**, then click **Apply** to clean and explore.")

internal_df = None
external_df = None
role_map: Dict[str, str] = {}
issues_report = {}

if up:
    raw_df = load_df(up)
    st.subheader("Schema Wizard: Map Your Columns")
    st.caption("We suggested matches below. Adjust any dropdowns where needed — unmapped roles are skipped.")

    cols_in_file = list(raw_df.columns)
    suggestions = suggest_mapping(cols_in_file)

    map_cols = {}
    grid = st.columns(3)
    for i, role in enumerate(STANDARD_ROLES):
        with grid[i % 3]:
            options = [""] + cols_in_file
            default = suggestions.get(role, "")
            idx = options.index(default) if default in options else 0
            sel = st.selectbox(role, options=options, index=idx, key=f"map_{role}")
            map_cols[role] = sel or None

    if st.button("Apply Mapping & Clean Data", type="primary"):
        internal_df, issues_report = apply_mapping_and_clean(raw_df, map_cols)
        role_map = map_cols
        st.success("Mapping applied — data cleaned.")
        st.json(issues_report)

# External wizard
ext_join_key = None
if up_ext:
    external_df = pd.DataFrame()
    try:
        external_df = load_df(up_ext)
    except Exception as e:
        st.warning(f"Could not read external file: {e}")
    if not external_df.empty:
        st.subheader("External Data Wizard")
        st.caption("Pick external join column and the internal key (zip/county/state) to enrich your data.")
        left, right = st.columns(2)
        with left:
            ext_key_col = st.selectbox("External join column", options=[""] + list(external_df.columns), index=0)
        with right:
            int_key_role = st.selectbox("Internal key (role)", options=["zip","county","state"], index=0)
        if ext_key_col:
            ext_join_key = (ext_key_col, int_key_role)

# If mapping not applied yet, stop until user clicks Apply
if up and internal_df is None:
    st.stop()

# -------------------------
# Data Quality & Metrics
# -------------------------
if internal_df is not None:
    st.subheader("Data Quality & Cleaning Report")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Rows (cleaned)", len(internal_df))
    with c2: st.metric("Standard-role columns", len(internal_df.columns))
    with c3: st.metric("Unmapped roles", len(issues_report.get("unmapped_roles", [])))
    if any(issues_report.values()):
        with st.expander("See details"):
            st.json(issues_report)

@st.cache_data(show_spinner=False)
def filter_df(df: pd.DataFrame, dept: str, date_field: str, start_date: datetime, end_date: datetime):
    out = df.copy()
    if dept and dept != "All Departments" and "department" in out.columns:
        out = out[out["department"] == dept]
    if date_field in out.columns:
        out = out[(out[date_field] >= pd.to_datetime(start_date)) & (out[date_field] <= pd.to_datetime(end_date))]
    return out

if internal_df is not None:
    work_df = filter_df(internal_df, dept, date_field, start_date, end_date)

    # Join external if configured
    if up_ext and ext_join_key is not None and 'external_df' in locals() and not external_df.empty:
        ext_col, int_role = ext_join_key
        if int_role in work_df.columns and ext_col in external_df.columns:
            tmp_ext = external_df.rename(columns={ext_col: int_role})
            work_df = work_df.merge(tmp_ext, on=int_role, how="left")

    st.subheader("Overview")
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Records in view", len(work_df))
    with m2:
        if "client_id" in work_df.columns:
            st.metric("Unique clients", work_df["client_id"].nunique())
    with m3:
        if "department" in work_df.columns:
            st.metric("Departments", work_df["department"].nunique())
    with m4:
        if "zip" in work_df.columns:
            st.metric("ZIPs covered", work_df["zip"].nunique())

    # -----------------
    # Charts
    # -----------------
    st.subheader("Trends & Breakdown")
    ch1, ch2 = st.columns(2)

    # Time series by month
    if date_field in work_df.columns:
        tmp = work_df[[date_field]].dropna().copy()
        if not tmp.empty:
            tmp["month"] = pd.to_datetime(tmp[date_field]).dt.to_period("M").dt.to_timestamp()
            ts = tmp.groupby("month").size().reset_index(name="count")
            with ch1:
                st.plotly_chart(px.line(ts, x="month", y="count", title=f"Records per month ({date_field})"), use_container_width=True)

    # Top legal issues or event types
    with ch2:
        if "legal_issue" in work_df.columns and work_df["legal_issue"].notna().any():
            top_issues = work_df["legal_issue"].fillna("Unknown").value_counts().head(10).reset_index()
            top_issues.columns = ["legal_issue","count"]
            st.plotly_chart(px.bar(top_issues, x="legal_issue", y="count", title="Top Legal Issues"), use_container_width=True)
        elif "event_type" in work_df.columns and work_df["event_type"].notna().any():
            top_ev = work_df["event_type"].fillna("Unknown").value_counts().head(10).reset_index()
            top_ev.columns = ["event_type","count"]
            st.plotly_chart(px.bar(top_ev, x="event_type", y="count", title="Top Outreach Event Types"), use_container_width=True)

    # Outcomes by department
    if "outcome" in work_df.columns and "department" in work_df.columns:
        st.subheader("Outcomes by Department")
        ob = work_df.groupby(["department","outcome"], dropna=False).size().reset_index(name="count")
        st.plotly_chart(px.bar(ob, x="department", y="count", color="outcome", barmode="stack", title="Outcome distribution"), use_container_width=True)

    # Demographics
    st.subheader("Demographics")
    d1, d2, d3 = st.columns(3)
    if "language" in work_df.columns and work_df["language"].notna().any():
        with d1:
            lang = work_df["language"].fillna("Unknown").value_counts().reset_index()
            lang.columns = ["language","count"]
            st.plotly_chart(px.pie(lang, names="language", values="count", title="Languages"), use_container_width=True)
    if "race_ethnicity" in work_df.columns and work_df["race_ethnicity"].notna().any():
        with d2:
            re_ = work_df["race_ethnicity"].fillna("Unknown").value_counts().reset_index()
            re_.columns = ["race_ethnicity","count"]
            st.plotly_chart(px.pie(re_, names="race_ethnicity", values="count", title="Race/Ethnicity"), use_container_width=True)
    if "income_pct_fpl" in work_df.columns and work_df["income_pct_fpl"].notna().any():
        with d3:
            bins = pd.cut(work_df["income_pct_fpl"], bins=[0,100,150,200,300,600], include_lowest=True)
            fpl = bins.value_counts().sort_index().reset_index()
            fpl.columns = ["FPL Band","count"]
            st.plotly_chart(px.bar(fpl, x="FPL Band", y="count", title="Income as % of FPL"), use_container_width=True)

    # -----------------
    # Maps
    # -----------------
    st.subheader("Maps")
    tabs = st.tabs(["Points (Cases/Events)", "ZIP Hotspots (Centroids)"])

    with tabs[0]:
        if {"latitude","longitude"}.issubset(work_df.columns) and work_df[["latitude","longitude"]].notna().any().any():
            pts = work_df.dropna(subset=["latitude","longitude"]) 
            if not pts.empty:
                m = folium.Map(location=[pts["latitude"].mean(), pts["longitude"].mean()], zoom_start=9)
                for _, r in pts.iterrows():
                    popup = folium.Popup(f"Dept: {r.get('department','')}<br>Issue: {r.get('legal_issue','')}<br>Date: {r.get(date_field,'')}", max_width=300)
                    folium.CircleMarker([r["latitude"], r["longitude"]], radius=4, popup=popup).add_to(m)
                st_html(m._repr_html_(), height=500)
        else:
            st.info("Add latitude/longitude columns for point maps.")

    with tabs[1]:
        if "zip" in work_df.columns and work_df["zip"].notna().any():
            zc = work_df.groupby("zip").size().reset_index(name="count")
            if {"latitude","longitude"}.issubset(work_df.columns) and work_df[["latitude","longitude"]].notna().any().any():
                cent = work_df.dropna(subset=["zip","latitude","longitude"]).groupby("zip")[['latitude','longitude']].mean().reset_index()
                zz = zc.merge(cent, on="zip", how="left").dropna(subset=["latitude","longitude"]) 
                if not zz.empty:
                    m2 = folium.Map(location=[zz["latitude"].mean(), zz["longitude"].mean()], zoom_start=8)
                    for _, r in zz.iterrows():
                        folium.Circle(
                            location=[r["latitude"], r["longitude"]],
                            radius=300,
                            popup=folium.Popup(f"ZIP {r['zip']}: {r['count']} records", max_width=200)
                        ).add_to(m2)
                    st_html(m2._repr_html_(), height=500)
                else:
                    st.info("Need some lat/long values to plot ZIP centroids.")
            else:
                st.dataframe(zc)
        else:
            st.info("No ZIPs found to aggregate.")

    # -----------------
    # Export
    # -----------------
    st.subheader("Export Filtered CSV")
    with io.StringIO() as buffer:
        work_df.to_csv(buffer, index=False)
        csv_bytes = buffer.getvalue().encode()
    st.download_button("Download filtered CSV", data=csv_bytes, file_name="legal_aid_filtered.csv", mime="text/csv")

# -------------------------
# Footer
# -------------------------
with st.expander("ℹ️ Setup & Tips"):
    st.markdown(
        """
        **Install & run (isolated env recommended)**  
        - `pip install -r requirements.txt`  
        - `streamlit run app.py`

        **Unknown columns?**  
        - Use the *Schema Wizard* to map headers to roles like `opened_date`, `zip`, `department`.  
        - Unmapped roles are safely ignored.

        **External data**  
        - Upload any CSV/Excel (e.g., Census indicators). Choose its join column (e.g., `ZipCode`) and the internal key (`zip`/`county`/`state`).

        **Maps**  
        - Provide `latitude` & `longitude` in your data for point maps.  
        - For boundary choropleths (ZIP/tract polygons), add GeoJSON support later.

        **Privacy**  
        - Runs locally or on Streamlit Cloud. Handle PII according to your policies.
        """
    )
