# app.py
# Streamlit Analytics App for a Nonprofit Legal Aid Organization — Schema Wizard Edition (PDF Export)
# -------------------------------------------------------------------------------------
# Run:  streamlit run app.py

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
from fpdf import FPDF  # For PDF export

# Optional fuzzy matcher
try:
    from rapidfuzz import process as fuzz_process
    HAS_FUZZ = True
except Exception:
    HAS_FUZZ = False

st.set_page_config(page_title="Legal Aid Analytics", page_icon="⚖️", layout="wide")

# -------------------------
# Constants
# -------------------------
DEPT_OPTIONS = [
    "All Departments","Family","Expungement","Consumer Law",
    "Tenant Rights & Housing","Administrative Law","Outreach",
    "Marketing","Language Access",
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
}

ZIP_RE = re.compile(r"^(\d{5})(?:-\d{4})?$")
PHONE_DIGITS_RE = re.compile(r"\D+")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False)
def load_df(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

@st.cache_data(show_spinner=False)
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

@st.cache_data(show_spinner=False)
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

# Cleaning
def to_date(s): return pd.to_datetime(s, errors="coerce")
def to_num(s, lo=None, hi=None):
    s = pd.to_numeric(s, errors="coerce")
    if lo is not None: s = s.where(s >= lo)
    if hi is not None: s = s.where(s <= hi)
    return s

def clean_zip(s):
    t = s.astype(str).str.replace(r"\D", "", regex=True)
    t = t.str[:5]
    has_val = t.str.len() > 0
    t = t.where(~has_val, t.str.zfill(5))
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

OPS = {
    "date": lambda s,c: to_date(s),
    "numeric": lambda s,c: to_num(s, c.get("min"), c.get("max")),
    "zip": lambda s,c: clean_zip(s),
    "phone": lambda s,c: clean_phone(s),
    "email": lambda s,c: clean_email(s),
}

def finalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "zip" in df.columns:
        df["zip"] = clean_zip(df["zip"]).astype("string")
    for col in ["client_id","phone","email","address","city","county","state"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    for col in ["opened_date","closed_date","event_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def apply_mapping_and_clean(df: pd.DataFrame, role_map: Dict[str,str]):
    df = df.copy()
    unified = pd.DataFrame(index=df.index)
    for role in STANDARD_ROLES:
        src = role_map.get(role)
        unified[role] = df[src] if (src and src in df.columns) else np.nan
    for role, cfg in DEFAULT_ROLE_CLEANING.items():
        if role in unified.columns:
            kind = cfg.get("type")
            if kind in OPS:
                unified[role] = OPS[kind](unified[role], cfg)
    return finalize_dtypes(unified)

# PDF Export
def export_pdf(df: pd.DataFrame) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    col_width = pdf.w / (len(df.columns) + 1)
    row_height = pdf.font_size * 1.5
    for col in df.columns:
        pdf.cell(col_width, row_height, str(col), border=1)
    pdf.ln(row_height)
    for _, row in df.iterrows():
        for item in row:
            pdf.cell(col_width, row_height, str(item), border=1)
        pdf.ln(row_height)
    return pdf.output(dest="S").encode("latin-1")

# -------------------------
# UI
# -------------------------
st.title("Legal Aid Analytics — Streamlit App")
st.write("Upload a file, map headers to roles with **Schema Wizard**, then explore.")

up = st.file_uploader("Upload CSV/Excel", type=["csv","xlsx"])
internal_df = None

if up:
    raw_df = load_df(up)
    st.subheader("Schema Wizard")
    cols_in_file = list(raw_df.columns)
    suggestions = suggest_mapping(cols_in_file)
    map_cols = {}
    grid = st.columns(3)
    for i, role in enumerate(STANDARD_ROLES):
        with grid[i % 3]:
            options = [""] + cols_in_file
            default = suggestions.get(role, "")
            sel = st.selectbox(role, options=options, index=options.index(default) if default in options else 0)
            map_cols[role] = sel or None
    if st.button("Apply Mapping & Clean Data", type="primary"):
        internal_df = apply_mapping_and_clean(raw_df, map_cols)
        st.success("Data cleaned.")

if internal_df is not None:
    st.subheader("Preview")
    st.dataframe(internal_df.head())

    st.subheader("Export to PDF")
    pdf_bytes = export_pdf(internal_df)
    st.download_button("Download PDF", data=pdf_bytes, file_name="legal_aid_data.pdf", mime="application/pdf")
