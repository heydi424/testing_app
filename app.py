# app.py
# Legal Aid Analytics — Automated Cleaning, Analysis, External Join + Q&A
# -----------------------------------------------------------------------
# Deploy: Streamlit Community Cloud
# runtime.txt  ->  python-3.11
# requirements ->  see bottom of this message

import io, re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import folium
from streamlit.components.v1 import html as st_html

# Optional fuzzy header suggestions
try:
    from rapidfuzz import process as fuzz_process
    HAS_FUZZ = True
except Exception:
    HAS_FUZZ = False

# Optional SQL for Q&A
try:
    import duckdb  # used only in "SQL mode" in Q&A
    HAS_DUCKDB = True
except Exception:
    HAS_DUCKDB = False

st.set_page_config(page_title="Legal Aid Analytics", page_icon="⚖️", layout="wide")

# -------------------------
# Role dictionary & cleaning rules
# -------------------------
STANDARD_ROLES = [
    "client_id","department","legal_issue","opened_date","closed_date",
    "zip","city","county","state","language","event_type","event_date",
    "outcome","referral_source","income_pct_fpl","age","race_ethnicity",
    "gender","phone","email","address","latitude","longitude"
]

ROLE_SYNONYMS = {
    "client_id": ["client id","clientid","case id","caseid","person_id","id"],
    "department": ["dept","unit","program","service_area","team"],
    "legal_issue": ["issue","case_type","matter","topic","problem","reason"],
    "opened_date": ["open date","intake_date","created","start_date","date_opened","filed","start"],
    "closed_date": ["close date","resolved","end_date","date_closed","closed","end"],
    "zip": ["zipcode","postal","postal_code","zip_code","zip5","code5","postal5"],
    "city": ["municipality","town"],
    "county": ["parish","borough","countyname","county_area"],
    "state": ["province","state_code","st","region","regioncode"],
    "language": ["primary_language","lang","primarylang"],
    "event_type": ["outreach_type","event kind","activity","activity_type"],
    "event_date": ["outreach_date","eventdate","activity_date","when","whenhappened"],
    "outcome": ["disposition","result","status","outcometext"],
    "referral_source": ["how heard","referrer","source","referred_by","heardvia","sourceinfo"],
    "income_pct_fpl": ["fpl","% fpl","income fpl","fplpct","income_pct_fpl"],
    "age": ["client_age","age_years","years"],
    "race_ethnicity": ["race","ethnicity","race/ethnicity","ethnicgroup"],
    "gender": ["sex","gender_identity","sexatbirth"],
    "phone": ["phone_number","mobile","cell","contact","contactnum"],
    "email": ["email_address","e-mail","mail","contactmail"],
    "address": ["street","addr","address1","address line 1","address_line_1","addr1","addr_line1","addr_line_1","addr_line"],
    "latitude": ["lat","y","ycoord"],
    "longitude": ["lon","lng","long","x","xcoord"],
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
        "family": "Family",
        "housing": "Tenant Rights & Housing", "tenant": "Tenant Rights & Housing",
        "expunge": "Expungement",
        "consumer": "Consumer Law",
        "admin": "Administrative Law",
        "outreach": "Outreach",
        "marketing": "Marketing",
        "language": "Language Access",
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

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def _ratio(x: pd.Series, pred) -> float:
    if x is None or x.empty: return 0.0
    try:
        return (x.dropna().map(pred)).mean()
    except Exception:
        return 0.0

def _is_date_series(s: pd.Series) -> float:
    return pd.to_datetime(s, errors="coerce").notna().mean()

def _is_zip_series(s: pd.Series) -> float:
    return s.astype(str).str.strip().str.extract(ZIP_RE)[0].notna().mean()

def _is_email_series(s: pd.Series) -> float:
    return s.astype(str).str.strip().str.lower().str.match(EMAIL_RE).fillna(False).mean()

def _is_phone_series(s: pd.Series) -> float:
    def ok(v):
        v = re.sub(r"\D+","", str(v))
        return (len(v)==10) or (len(v)==11 and v.startswith("1"))
    return _ratio(s, ok)

def _is_lat_series(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return s.between(-90, 90).mean()

def _is_lon_series(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return s.between(-180, 180).mean()

def _is_categorical(s: pd.Series) -> float:
    n = len(s.dropna())
    if n == 0: return 0.0
    uniq = s.dropna().nunique()
    return 1.0 - min(1.0, uniq / max(1, n))

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

@st.cache_data(show_spinner=False)
def suggest_mapping_by_header(columns: List[str]) -> Dict[str, Optional[str]]:
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
        # fuzzy
        if not role in suggestions and HAS_FUZZ:
            best = fuzz_process.extractOne(role, list(norm_cols.values()))
            if best and best[1] >= 88:
                for c, n in norm_cols.items():
                    if n == best[0]:
                        suggestions[role] = c; break
    return suggestions

def choose_best_by_content(df: pd.DataFrame, roles_missing: List[str], used: set) -> Dict[str, Optional[str]]:
    res = {}
    cols = [c for c in df.columns if c not in used]
    norm = {c: _norm(c) for c in cols}

    def best(col_scores):
        if not col_scores: return None
        col_scores.sort(key=lambda x: x[1], reverse=True)
        top, score = col_scores[0]
        return top if score >= 0.55 else None

    date_name_rank = {
        "opened_date": ["open","intake","start","created","filed"],
        "closed_date": ["close","resolved","end","completed"],
        "event_date":  ["event","outreach","clinic","workshop","session","activity"],
    }

    for role in roles_missing:
        candidates = []
        if role in ("opened_date","closed_date","event_date"):
            for c in cols:
                score = _is_date_series(df[c]) + (0.1 if any(k in norm[c] for k in date_name_rank[role]) else 0.0)
                candidates.append((c, score))
            pick = best(candidates)
        elif role == "zip":
            pick = best([(c, _is_zip_series(df[c]) + (0.1 if "zip" in norm[c] or "postal" in norm[c] else 0.0)) for c in cols])
        elif role == "email":
            pick = best([(c, _is_email_series(df[c]) + (0.1 if "mail" in norm[c] else 0.0)) for c in cols])
        elif role == "phone":
            pick = best([(c, _is_phone_series(df[c]) + (0.1 if "phone" in norm[c] or "mobile" in norm[c] else 0.0)) for c in cols])
        elif role == "latitude":
            pick = best([(c, _is_lat_series(df[c]) + (0.1 if "lat" in norm[c] or norm[c] == "y" else 0.0)) for c in cols])
        elif role == "longitude":
            pick = best([(c, _is_lon_series(df[c]) + (0.1 if "lon" in norm[c] or "lng" in norm[c] or norm[c] == "x" else 0.0)) for c in cols])
        elif role == "department":
            pick = best([(c, (0.1 if any(k in norm[c] for k in ["dept","unit","program","department","team"]) else 0.0) + 0.6*_is_categorical(df[c])) for c in cols])
        elif role in ("legal_issue","event_type","outcome","referral_source","language","race_ethnicity","gender","city","county","state","address"):
            hints = {
                "legal_issue": ["issue","case","matter","topic","problem","reason"],
                "event_type": ["event","type","outreach","clinic","workshop","activity"],
                "outcome": ["outcome","result","disposition","status"],
                "referral_source": ["referral","referred","source","heard"],
                "language": ["language","lang"],
                "race_ethnicity": ["race","ethnicity"],
                "gender": ["gender","sex"],
                "city": ["city","municipality","town"],
                "county": ["county","parish","borough"],
                "state": ["state","province","region"],
                "address": ["address","street","addr"],
            }
            pick = best([(c, (0.2 if any(k in norm[c] for k in hints[role]) else 0.0) + 0.6*_is_categorical(df[c])) for c in cols])
        elif role in ("age","income_pct_fpl"):
            pick = best([(c, 0.6*(_is_categorical(df[c]) < 0.7)) for c in cols])
        else:
            pick = None

        if pick and pick not in used:
            res[role] = pick; used.add(pick)
        else:
            res[role] = None

    return res

def auto_map_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    by_header = suggest_mapping_by_header(list(df.columns))
    used = {c for c in by_header.values() if c}
    missing = [r for r in STANDARD_ROLES if not by_header.get(r)]
    by_content = choose_best_by_content(df, missing, used)
    return {r: by_header.get(r) or by_content.get(r) for r in STANDARD_ROLES}

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
        digits = re.sub(r"\D+","", str(x))
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

def apply_mapping_and_clean(df: pd.DataFrame, role_map: Dict[str,Optional[str]], cleaning=DEFAULT_ROLE_CLEANING):
    src = df.copy()
    unified = {}
    issues = {}

    for role in STANDARD_ROLES:
        col = role_map.get(role)
        unified[role] = src[col] if col and col in src.columns else pd.Series([np.nan]*len(src), index=src.index)

    for role, cfg in cleaning.items():
        if role in unified:
            before = pd.isna(unified[role]).sum()
            kind = cfg.get("type")
            if kind in OPS:
                unified[role] = OPS[kind](unified[role], cfg)
            after = pd.isna(unified[role]).sum()
            if after > before:
                issues.setdefault("values_nullified", []).append(f"{role}: {after-before}")

    cleaned = src.copy()
    for role, series in unified.items():
        cleaned[role] = series  # add standardized role columns

    subset = [c for c in ["client_id","department","opened_date","legal_issue","zip"] if c in cleaned.columns]
    if subset:
        before_len = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=subset, keep="first")
        removed = before_len - len(cleaned)
        if removed: issues.setdefault("duplicates_removed", []).append(str(removed))

    issues["unmapped_roles"] = [r for r in STANDARD_ROLES if not role_map.get(r)]
    return cleaned, issues

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("⚖️ Legal Aid Analytics")
st.sidebar.caption("Upload → Auto-clean → (optional) Join → Analyze → Ask → Export")

up = st.sidebar.file_uploader("Internal CSV/Excel", type=["csv","xlsx"], accept_multiple_files=False)
up_ext = st.sidebar.file_uploader("External CSV/Excel (join on zip/county/state)", type=["csv","xlsx"], accept_multiple_files=False)

st.title("Legal Aid Analytics — Automated Data Science Workflow")
if not up:
    st.info("👆 Upload a CSV or Excel file to begin.")
    st.stop()

raw_df = load_df(up)
role_map = auto_map_columns(raw_df)
clean_df, issues = apply_mapping_and_clean(raw_df, role_map)

# Optional external join
work_df = clean_df.copy()
if up_ext:
    ext_df = load_df(up_ext)
    st.subheader("External Data Join")
    c1, c2 = st.columns(2)
    with c1:
        ext_key = st.selectbox("External join column", options=list(ext_df.columns), index=0)
    with c2:
        int_key = st.selectbox("Internal key (role)", options=[k for k in ["zip","county","state"] if k in work_df.columns], index=0 if "zip" in work_df.columns else 0)
    if ext_key and int_key:
        work_df = work_df.merge(ext_df.rename(columns={ext_key: int_key}), on=int_key, how="left")
        st.success(f"Joined on `{int_key}`.")

# -------------------------
# Filters (only if fields exist)
# -------------------------
st.subheader("Filters")
available_dates = [c for c in ["opened_date","event_date","closed_date"] if c in work_df.columns and not work_df[c].isna().all()]
date_field = st.selectbox("Date field", options=available_dates or ["(none)"])
if date_field != "(none)":
    min_d = pd.to_datetime(work_df[date_field]).min()
    max_d = pd.to_datetime(work_df[date_field]).max()
    default_start = (max_d - pd.DateOffset(years=1)).date() if pd.notna(max_d) else datetime(2023,1,1).date()
    start_date, end_date = st.date_input("Date range", value=(default_start, (max_d.date() if pd.notna(max_d) else datetime.today().date())))
else:
    start_date, end_date = None, None

cat_candidates = [c for c in ["department","legal_issue","event_type","outcome","language","state","county","city","zip"] if c in work_df.columns]
filter_values = {}
if cat_candidates:
    cols = st.columns(min(4, len(cat_candidates)))
    for i, c in enumerate(cat_candidates):
        with cols[i % len(cols)]:
            opts = ["(All)"] + sorted([str(x) for x in work_df[c].dropna().unique()][:500])
            sel = st.multiselect(c, options=opts, default=["(All)"])
            filter_values[c] = sel

f_df = work_df.copy()
if date_field != "(none)" and start_date and end_date:
    f_df = f_df[(pd.to_datetime(f_df[date_field]) >= pd.to_datetime(start_date)) &
                (pd.to_datetime(f_df[date_field]) <= pd.to_datetime(end_date))]
for col, choices in filter_values.items():
    if choices and "(All)" not in choices:
        f_df = f_df[f_df[col].astype(str).isin(set(map(str, choices)))]

# -------------------------
# Data quality & mapping
# -------------------------
st.subheader("Data Quality & Mapping")
m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Rows (cleaned)", len(clean_df))
with m2: st.metric("Rows in view", len(f_df))
with m3: st.metric("Mapped roles", sum(1 for r in STANDARD_ROLES if role_map.get(r)))
with m4: st.metric("Unmapped roles", len([r for r in STANDARD_ROLES if not role_map.get(r)]))
with st.expander("Detected mapping"): st.json({r: role_map.get(r) for r in STANDARD_ROLES if role_map.get(r)})
with st.expander("Cleaning report"): st.json(issues)

# -------------------------
# Overview & visuals
# -------------------------
st.subheader("Overview")
k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Unique clients", f_df["client_id"].nunique() if "client_id" in f_df.columns else 0)
with k2: st.metric("Departments", f_df["department"].nunique() if "department" in f_df.columns else 0)
with k3: st.metric("ZIPs covered", f_df["zip"].nunique() if "zip" in f_df.columns else 0)
with k4:
    if date_field != "(none)":
        st.metric("Date span", f"{pd.to_datetime(f_df[date_field]).min().date()} → {pd.to_datetime(f_df[date_field]).max().date()}" if not f_df.empty else "—")

st.subheader("Trends & Breakdown")
c1, c2 = st.columns(2)
if date_field != "(none)" and date_field in f_df.columns and not f_df[date_field].isna().all():
    tmp = f_df[[date_field]].dropna().copy()
    if not tmp.empty:
        tmp["month"] = pd.to_datetime(tmp[date_field]).dt.to_period("M").dt.to_timestamp()
        ts = tmp.groupby("month").size().reset_index(name="count")
        with c1:
            st.plotly_chart(px.line(ts, x="month", y="count", title=f"Records per month ({date_field})"), use_container_width=True)

with c2:
    if "legal_issue" in f_df.columns and f_df["legal_issue"].notna().any():
        top_issues = f_df["legal_issue"].fillna("Unknown").value_counts().head(10).reset_index()
        top_issues.columns = ["legal_issue","count"]
        st.plotly_chart(px.bar(top_issues, x="legal_issue", y="count", title="Top Legal Issues"), use_container_width=True)
    elif "event_type" in f_df.columns and f_df["event_type"].notna().any():
        top_ev = f_df["event_type"].fillna("Unknown").value_counts().head(10).reset_index()
        top_ev.columns = ["event_type","count"]
        st.plotly_chart(px.bar(top_ev, x="event_type", y="count", title="Top Outreach Event Types"), use_container_width=True)

if {"outcome","department"}.issubset(f_df.columns):
    st.plotly_chart(
        px.bar(
            f_df.groupby(["department","outcome"], dropna=False).size().reset_index(name="count"),
            x="department", y="count", color="outcome", barmode="stack", title="Outcome distribution by department"
        ),
        use_container_width=True
    )

st.subheader("Demographics")
d1, d2, d3, d4 = st.columns(4)
if "language" in f_df.columns and f_df["language"].notna().any():
    with d1:
        lang = f_df["language"].fillna("Unknown").value_counts().reset_index()
        lang.columns = ["language","count"]
        st.plotly_chart(px.pie(lang, names="language", values="count", title="Languages"), use_container_width=True)
if "race_ethnicity" in f_df.columns and f_df["race_ethnicity"].notna().any():
    with d2:
        re_ = f_df["race_ethnicity"].fillna("Unknown").value_counts().reset_index()
        re_.columns = ["race_ethnicity","count"]
        st.plotly_chart(px.pie(re_, names="race_ethnicity", values="count", title="Race/Ethnicity"), use_container_width=True)
if "gender" in f_df.columns and f_df["gender"].notna().any():
    with d3:
        g = f_df["gender"].fillna("Unknown").value_counts().reset_index()
        g.columns = ["gender","count"]
        st.plotly_chart(px.pie(g, names="gender", values="count", title="Gender"), use_container_width=True)
if "age" in f_df.columns and f_df["age"].notna().any():
    with d4:
        st.plotly_chart(px.histogram(f_df, x="age", nbins=20, title="Age distribution"), use_container_width=True)

if "income_pct_fpl" in f_df.columns and f_df["income_pct_fpl"].notna().any():
    bins = pd.cut(f_df["income_pct_fpl"], bins=[0,100,150,200,300,600], include_lowest=True)
    fpl = bins.value_counts().sort_index().reset_index()
    fpl.columns = ["FPL Band","count"]
    st.plotly_chart(px.bar(fpl, x="FPL Band", y="count", title="Income as % of FPL"), use_container_width=True)

# -------------------------
# Maps
# -------------------------
st.subheader("Maps")
tabs = st.tabs(["Points (lat/long)", "ZIP hotspot table"])
with tabs[0]:
    if {"latitude","longitude"}.issubset(f_df.columns) and f_df[["latitude","longitude"]].notna().any().any():
        pts = f_df.dropna(subset=["latitude","longitude"])
        if not pts.empty:
            m = folium.Map(location=[pts["latitude"].mean(), pts["longitude"].mean()], zoom_start=9)
            for _, r in pts.iterrows():
                popup = folium.Popup(
                    f"Dept: {r.get('department','')}<br>"
                    f"Issue: {r.get('legal_issue','')}<br>"
                    f"Date: {r.get(date_field,'') if date_field and date_field!='(none)' else ''}",
                    max_width=280
                )
                folium.CircleMarker([r["latitude"], r["longitude"]], radius=4, popup=popup).add_to(m)
            st_html(m._repr_html_(), height=520)
    else:
        st.info("Add `latitude` & `longitude` columns to see a point map.")
with tabs[1]:
    if "zip" in f_df.columns and f_df["zip"].notna().any():
        zc = f_df.groupby("zip").size().reset_index(name="count").sort_values("count", ascending=False)
        st.dataframe(zc, use_container_width=True)
    else:
        st.info("No ZIP column found.")

# -------------------------
# Ask a question (NL or SQL)
# -------------------------
st.subheader("Ask a question about the data")
st.caption("Examples (natural language): 'count by department', 'top 5 legal_issue', 'average age by department', 'rows where state=NY'. "
           "Set **SQL mode** to run a DuckDB SQL query like: `select state, count(*) from df group by 1 order by 2 desc limit 10`.")

q1, q2 = st.columns([3,1])
question = q1.text_input("Your question", value="")
sql_mode = q2.toggle("SQL mode (DuckDB)", value=False, help="Run raw SQL against the filtered dataframe")
run_q = st.button("Run")

def answer_nl(q: str, df: pd.DataFrame):
    q = q.strip().lower()
    if not q: return None, "Type a question."
    # top N of column
    m = re.match(r"top\s+(\d+)\s+([a-z0-9_]+)", q)
    if m:
        n = int(m.group(1)); col = m.group(2)
        if col in df.columns:
            out = df[col].value_counts().head(n).reset_index()
            out.columns = [col, "count"]
            return out, f"Top {n} of {col}"
    # count by column
    m = re.match(r"(count|how many)\s+(by|per)\s+([a-z0-9_]+)", q)
    if m:
        col = m.group(3)
        if col in df.columns:
            out = df.groupby(col).size().reset_index(name="count").sort_values("count", ascending=False)
            return out, f"Count by {col}"
    # average/mean/median of X by Y
    m = re.match(r"(average|mean|median)\s+([a-z0-9_]+)\s+(by|per)\s+([a-z0-9_]+)", q)
    if m:
        agg = m.group(1); x = m.group(2); y = m.group(4)
        if x in df.columns and y in df.columns:
            func = "mean" if agg in ("average","mean") else "median"
            out = df.groupby(y)[x].agg(func).reset_index().sort_values(x, ascending=False)
            return out, f"{agg.title()} {x} by {y}"
    # sum of X by Y
    m = re.match(r"(sum|total)\s+([a-z0-9_]+)\s+(by|per)\s+([a-z0-9_]+)", q)
    if m:
        x = m.group(2); y = m.group(4)
        if x in df.columns and y in df.columns:
            out = df.groupby(y)[x].sum(min_count=1).reset_index().sort_values(x, ascending=False)
            return out, f"Sum {x} by {y}"
    # rows where col=value (supports = and ==)
    m = re.match(r"(rows|show)\s+where\s+([a-z0-9_]+)\s*={1,2}\s*([a-z0-9_\-@.]+)", q)
    if m:
        col = m.group(2); val = m.group(3)
        if col in df.columns:
            out = df[df[col].astype(str).str.lower() == val.lower()]
            return out.head(100), f"Rows where {col} == {val} (showing up to 100)"
    # how many rows
    if q in ("how many rows","row count","count rows"):
        return pd.DataFrame({"rows":[len(df)]}), "Row count"
    return None, "Sorry — try a pattern like: 'count by department', 'top 10 zip', 'average age by department', or switch to SQL mode."

if run_q:
    if sql_mode:
        if not HAS_DUCKDB:
            st.error("DuckDB not available. Add `duckdb` to requirements.txt or turn off SQL mode.")
        else:
            try:
                # register dataframe as a DuckDB view
                duckdb.sql("DROP VIEW IF EXISTS df")
                duckdb.register("df", f_df)
                res = duckdb.sql(question).df()
                st.dataframe(res, use_container_width=True)
            except Exception as e:
                st.error(f"SQL error: {e}")
    else:
        res, msg = answer_nl(question, f_df)
        st.caption(msg)
        if isinstance(res, pd.DataFrame):
            st.dataframe(res, use_container_width=True)

# -------------------------
# Auto-generated summary report
# -------------------------
st.subheader("Analysis Report (Markdown)")
def build_report(df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"# Legal Aid Analytics — Automated Report")
    lines.append(f"**Rows in view:** {len(df)}")
    if "client_id" in df.columns: lines.append(f"**Unique clients:** {df['client_id'].nunique()}")
    if "department" in df.columns: lines.append(f"**Departments:** {df['department'].nunique()}")
    if "zip" in df.columns: lines.append(f"**ZIPs covered:** {df['zip'].nunique()}")
    for dcol in ["opened_date","event_date","closed_date"]:
        if dcol in df.columns and not df[dcol].isna().all():
            rng = f"{pd.to_datetime(df[dcol]).min().date()} → {pd.to_datetime(df[dcol]).max().date()}"
            lines.append(f"**{dcol} span:** {rng}")
    if "legal_issue" in df.columns and df["legal_issue"].notna().any():
        top = df["legal_issue"].value_counts().head(5)
        lines.append("**Top legal issues:**"); lines += [f"- {k}: {v}" for k, v in top.items()]
    elif "event_type" in df.columns and df["event_type"].notna().any():
        top = df["event_type"].value_counts().head(5)
        lines.append("**Top outreach event types:**"); lines += [f"- {k}: {v}" for k, v in top.items()]
    if {"department","outcome"}.issubset(df.columns):
        ob = df.groupby(["department","outcome"]).size().reset_index(name="count").sort_values("count", ascending=False).head(10)
        lines.append("**Outcomes by department (top):**")
        for _, r in ob.iterrows(): lines.append(f"- {r['department']}: {r['outcome']} — {r['count']}")
    for col, label in [("language","Languages"), ("race_ethnicity","Race/Ethnicity"), ("gender","Gender")]:
        if col in df.columns and df[col].notna().any():
            top = df[col].value_counts().head(5)
            lines.append(f"**{label} (top):**"); lines += [f"- {k}: {v}" for k, v in top.items()]
    if "income_pct_fpl" in df.columns and df["income_pct_fpl"].notna().any():
        bands = pd.cut(df["income_pct_fpl"], bins=[0,100,150,200,300,600], include_lowest=True).value_counts().sort_index()
        lines.append("**Income (FPL bands):**"); lines += [f"- {k}: {v}" for k, v in bands.items()]
    if "age" in df.columns and df["age"].notna().any():
        lines.append(f"**Age:** mean={df['age'].mean():.1f}, median={df['age'].median():.1f}")
    return "\n".join(lines)

report_md = build_report(f_df)
st.markdown(report_md)
st.download_button("⬇️ Download report (.md)", report_md.encode(), file_name="legal_aid_report.md", mime="text/markdown")

# -------------------------
# Export filtered data
# -------------------------
st.subheader("Export")
with io.StringIO() as buffer:
    f_df.to_csv(buffer, index=False)
    csv_bytes = buffer.getvalue().encode()
st.download_button("Download filtered CSV", data=csv_bytes, file_name="legal_aid_filtered.csv", mime="text/csv")

with st.expander("ℹ️ Notes"):
    st.markdown("""
- Columns are **auto-detected** from headers and the **data itself** (dates, ZIPs, lat/long, emails, phones).
- Ask questions in **plain language** (e.g., *count by department*, *top 10 zip*, *average age by department*), or switch to **SQL mode** to run DuckDB queries.
- For maps you need `latitude` & `longitude`. If you only have ZIP, you’ll still get a ZIP hotspot table.
- Join an external file on `zip`, `county`, or `state` to enrich your analysis.
""")
