# app.py
# Legal Aid Analytics — Streamlit App with Schema Wizard & Auto Cleaning

import io
import re
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import folium
from streamlit.components.v1 import html as st_html

try:
    from rapidfuzz import process as fuzz_process
    HAS_FUZZ = True
except ImportError:
    HAS_FUZZ = False

st.set_page_config(page_title="Legal Aid Analytics", page_icon="⚖️", layout="wide")

# -------------------------
# Constants
# -------------------------
DEPT_OPTIONS = [
    "All Departments", "Family", "Expungement", "Consumer Law",
    "Tenant Rights & Housing", "Administrative Law", "Outreach",
    "Marketing", "Language Access"
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
PHONE_DIGITS_RE = re.compile(r"\D+")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# -------------------------
# Loaders
# -------------------------
@st.cache_data
def load_df(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def suggest_mapping(columns: List[str]):
    suggestions = {}
    norm_cols = {c: _norm(c) for c in columns}
    for role in STANDARD_ROLES:
        for c, n in norm_cols.items():
            if n == _norm(role) or n.replace(" ", "_") == role:
                suggestions[role] = c; break
        if role in suggestions: continue
        for syn in ROLE_SYNONYMS.get(role, []):
            for c, n in norm_cols.items():
                if n == _norm(syn):
                    suggestions[role] = c; break
            if role in suggestions: break
        if role in suggestions or not HAS_FUZZ: continue
        best = fuzz_process.extractOne(role, list(norm_cols.values()))
        if best and best[1] >= 85:
            for c, n in norm_cols.items():
                if n == best[0]:
                    suggestions[role] = c; break
    return suggestions

# -------------------------
# Cleaning helpers
# -------------------------
def to_date(s): return pd.to_datetime(s, errors="coerce")

def to_num(s, lo=None, hi=None):
    s = pd.to_numeric(s, errors="coerce")
    if lo is not None: s = s.where(s >= lo)
    if hi is not None: s = s.where(s <= hi)
    return s

def clean_zip(s):
    t = s.astype(str).str.replace(r"\D", "", regex=True).str[:5]
    t = t.where(t.str.len() > 0, np.nan)
    t = t.str.zfill(5)
    t = t.where(t.str.fullmatch(r"\d{5}").fillna(False), np.nan)
    return t

def clean_phone(s):
    def f(x):
        if pd.isna(x): return np.nan
        digits = PHONE_DIGITS_RE.sub("", str(x))
        if len(digits) == 10:
            return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"
        if len(digits) == 11 and digits.startswith("1"):
            d = digits[1:]
            return f"+1 ({d[0:3]}) {d[3:6]}-{d[6:10]}"
        return np.nan
    return s.apply(f)

def clean_email(s):
    return s.astype(str).str.strip().str.lower().where(lambda x: x.str.match(EMAIL_RE), np.nan)

def case_map(s, mapping=None, upper=False, title=False, standardize_case=False):
    if mapping:
        return s.apply(lambda x: mapping.get(str(x).strip().lower(), str(x).strip()) if pd.notna(x) else np.nan)
    if upper: return s.apply(lambda x: str(x).strip().upper() if pd.notna(x) else np.nan)
    if title: return s.apply(lambda x: str(x).strip().title() if pd.notna(x) else np.nan)
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

# -------------------------
# Apply mapping & clean
# -------------------------
def finalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "zip" in df.columns:
        df["zip"] = clean_zip(df["zip"]).astype("string")
    string_cols = [
        "client_id","phone","email","address","city","county","state",
        "language","department","legal_issue","event_type","outcome",
        "referral_source","race_ethnicity","gender"
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")
    for col in ["opened_date","closed_date","event_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    bounds = {
        "income_pct_fpl": (0, 600),
        "age": (0, 120),
        "latitude": (-90, 90),
        "longitude": (-180, 180),
    }
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if lo is not None: df[col] = df[col].where(df[col] >= lo)
            if hi is not None: df[col] = df[col].where(df[col] <= hi)
    for col in [
        "department","language","city","county","state","legal_issue",
        "event_type","outcome","referral_source","race_ethnicity","gender"
    ]:
        if col in df.columns and df[col].nunique(dropna=True) <= 1000:
            df[col] = df[col].astype("category")
    return df

def apply_mapping_and_clean(df: pd.DataFrame, role_map: Dict[str,str]):
    df = df.copy()
    unified = pd.DataFrame(index=df.index)
    for role in STANDARD_ROLES:
        src = role_map.get(role)
        unified[role] = df[src] if (src and src in df.columns) else np.nan
    issues = {}
    for role, cfg in DEFAULT_ROLE_CLEANING.items():
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
    return finalize_dtypes(unified), issues

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("⚖️ Legal Aid Analytics")
up = st.sidebar.file_uploader("Upload Internal CSV/Excel", type=["csv","xlsx"])
up_ext = st.sidebar.file_uploader("Upload External CSV/Excel", type=["csv","xlsx"])
dept = st.sidebar.selectbox("Department", options=DEPT_OPTIONS, index=0)
date_field = st.sidebar.selectbox("Date field", options=["opened_date","event_date","closed_date"])
start_date = st.sidebar.date_input("Start", value=pd.to_datetime("2023-01-01").date())
end_date = st.sidebar.date_input("End", value=pd.to_datetime("today").date())

# -------------------------
# Main
# -------------------------
st.title("Legal Aid Analytics — Streamlit App")
internal_df = None
issues_report = {}

if up:
    raw_df = load_df(up)
    st.subheader("Schema Wizard")
    cols_in_file = list(raw_df.columns)
    suggestions = suggest_mapping(cols_in_file)
    map_cols = {}
    grid = st.columns(3)
    for i, role in enumerate(STANDARD_ROLES):
        with grid[i % 3]:
            opts = [""] + cols_in_file
            default = suggestions.get(role, "")
            idx = opts.index(default) if default in opts else 0
            sel = st.selectbox(role, options=opts, index=idx, key=f"map_{role}")
            map_cols[role] = sel or None
    if st.button("Apply Mapping & Clean Data", type="primary"):
        internal_df, issues_report = apply_mapping_and_clean(raw_df, map_cols)
        st.success("Mapping applied — data cleaned.")
        st.json(issues_report)

if up and internal_df is None:
    st.stop()

def filter_df(df: pd.DataFrame):
    out = df.copy()
    if dept != "All Departments" and "department" in out.columns:
        out = out[out["department"].astype(str) == dept]
    if date_field in out.columns:
        dates = pd.to_datetime(out[date_field], errors="coerce")
        if dates.notna().any():
            mask = dates.between(pd.to_datetime(start_date), pd.to_datetime(end_date))
            out = out[mask]
    return out

if internal_df is not None:
    work_df = filter_df(internal_df)
    if up_ext:
        ext_df = load_df(up_ext)
        ext_col = st.selectbox("External join column", options=[""] + list(ext_df.columns))
        int_role = st.selectbox("Internal key", options=["zip","county","state"])
        if ext_col:
            ext_df = ext_df.rename(columns={ext_col: int_role})
            work_df = work_df.merge(ext_df, on=int_role, how="left")

    # Metrics
    st.metric("Records in view", len(work_df))
    if "client_id" in work_df.columns:
        st.metric("Unique clients", work_df["client_id"].nunique())
    if "zip" in work_df.columns:
        st.metric("ZIPs covered", work_df["zip"].nunique())

    # Charts
    if date_field in work_df.columns and work_df[date_field].notna().any():
        tmp = work_df.copy()
        tmp["month"] = pd.to_datetime(tmp[date_field]).dt.to_period("M").dt.to_timestamp()
        ts = tmp.groupby("month").size().reset_index(name="count")
        st.plotly_chart(px.line(ts, x="month", y="count", title="Records per month"))

    if "legal_issue" in work_df.columns:
        top = work_df["legal_issue"].fillna("Unknown").value_counts().head(10).reset_index()
        top.columns = ["legal_issue","count"]
        st.plotly_chart(px.bar(top, x="legal_issue", y="count", title="Top Legal Issues"))

    # Maps
    if {"latitude","longitude"}.issubset(work_df.columns) and work_df[["latitude","longitude"]].notna().any().any():
        pts = work_df.dropna(subset=["latitude","longitude"])
        m = folium.Map(location=[pts["latitude"].mean(), pts["longitude"].mean()], zoom_start=9)
        for _, r in pts.iterrows():
            folium.CircleMarker([r["latitude"], r["longitude"]], radius=4).add_to(m)
        st_html(m._repr_html_(), height=500)
    elif "zip" in work_df.columns:
        z_df = work_df.copy()
        z_df["zip"] = clean_zip(z_df["zip"])
        z_df = z_df.dropna(subset=["zip"])
        st.dataframe(z_df.groupby("zip").size().reset_index(name="count"))

    # Export
    st.download_button("Download filtered CSV", data=work_df.to_csv(index=False), file_name="filtered.csv")
