# utils.py - shared helpers for the SYNLAB Nigeria market-intelligence dashboard
import base64
import os

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "synlab_dashboard.csv")
ASSETS = os.path.join(BASE_DIR, "assets")

# SYNLAB blue-only palette
CERULEAN, MIDNIGHT, HALFBAKED, NAVY = "#0077AD", "#003765", "#7CB8D3", "#0A2647"
BLUE_MED, BLUE_LIGHT, SLATE = "#205295", "#2C8FC7", "#64748B"
PALETTE = [MIDNIGHT, CERULEAN, BLUE_MED, BLUE_LIGHT, HALFBAKED, NAVY]
INTENT_ORDER = ["Definitely would", "Probably would", "Might or might not", "Probably would not",
                "Definitely would not", "Cannot judge from the information shown"]
INTENT_COLORS = [MIDNIGHT, CERULEAN, HALFBAKED, "#B8D4E3", "#DCE9F1", "#EEF2F5"]
AGE_ORDER = ["18-24", "25-34", "35-44", "45-54", "55-64", "65 or above"]
MIN_N = 15  # smallest group shown in a cell, card or ranking


# ------------------------------------------------------------------ logo
def get_logo_b64(logo_filename="synlab_logo.png"):
    path = os.path.join(ASSETS, logo_filename)
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")
    return None


def render_hero_logo(height=65, style="margin-bottom: 12px;", logo="synlab_logo.png"):
    b64 = get_logo_b64(logo)
    if b64:
        return (f'<img src="data:image/png;base64,{b64}" style="height:{height}px; {style} '
                f'filter: drop-shadow(0px 2px 4px rgba(0,0,0,0.15));" />')
    return '<span style="font-size: 48px;">&#128300;</span>'


# ------------------------------------------------------------------ page setup
def inject_css():
    css_path = os.path.join(ASSETS, "app.css")
    css = ""
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as fh:
            css = fh.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def setup_page(title):
    try:
        from PIL import Image
        icon = Image.open(os.path.join(ASSETS, "synlab_logo.png"))
    except Exception:
        icon = "🔬"
    st.set_page_config(page_title=f"{title} | SYNLAB Nigeria", page_icon=icon, layout="wide",
                       initial_sidebar_state="collapsed")
    inject_css()


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()
    return df


# ------------------------------------------------------------------ filters
def filter_bar(df, extended=False):
    """Render the filter bar and return the filtered frame."""
    ss = st.session_state
    regions = sorted(df["region"].dropna().unique().tolist())

    st.markdown('<div class="filter-label">Filters</div>', unsafe_allow_html=True)
    c = st.columns([1.1, 1.3, 1.1, 1.3, 1.2]) if extended else st.columns([1.3, 3.4])

    with c[0]:
        sel_region = st.multiselect("Region", regions, default=regions, key="f_region")
    out = df[df["region"].isin(sel_region)] if sel_region else df

    if extended:
        lga_opts = sorted(out["lga"].dropna().unique().tolist())
        ss["f_lga"] = [x for x in ss.get("f_lga", []) if x in lga_opts]
        with c[1]:
            sel_lga = st.multiselect("LGA / Area Council", lga_opts, key="f_lga")
        with c[2]:
            sel_age = st.multiselect("Age group", [a for a in AGE_ORDER if a in set(df["age"].dropna())], key="f_age")
        with c[3]:
            emp_opts = sorted(df["employment"].dropna().unique().tolist())
            sel_emp = st.multiselect("Employment", emp_opts, key="f_emp")
        with c[4]:
            ins_opts = [x for x in ["Insured", "Uninsured", "Not sure / no answer"] if x in set(df["insurance"].dropna())]
            sel_ins = st.multiselect("Insurance", ins_opts, key="f_ins")
        if sel_lga: out = out[out["lga"].isin(sel_lga)]
        if sel_age: out = out[out["age"].isin(sel_age)]
        if sel_emp: out = out[out["employment"].isin(sel_emp)]
        if sel_ins: out = out[out["insurance"].isin(sel_ins)]
    return out


@st.cache_data
def load_summary():
    """Respondent-base counts written by prepare_data.py (total, eligible, full survey)."""
    import json
    path = os.path.join(BASE_DIR, "data", "survey_summary.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


# ------------------------------------------------------------------ calculations
def full(df):
    """Respondents who passed the digital-access gate and answered the full questionnaire."""
    return df[df["full_survey"] == 1]


def pct_of(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean() * 100) if len(s) else float("nan")


def fmt(x, d=0, suffix="%"):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}{suffix}"


def flag_share(df, prefix):
    """% of respondents ticking each option of a multi-select block, plus the base size."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return pd.Series(dtype=float), 0
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    base = sub.dropna(how="all")
    if base.empty:
        return pd.Series(dtype=float), 0
    shares = base.mean() * 100
    shares.index = [c[len(prefix):] for c in cols]
    return shares.sort_values(ascending=False), len(base)


def cat_share(series, order=None):
    s = series.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    out = s.value_counts(normalize=True) * 100
    if order:
        out = out.reindex([o for o in order if o in out.index])
    return out


def peso(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"₦{x:,.0f}"


def tier_label(price):
    return f"₦{int(price):,}"


def funnel_counts(df):
    """Funnel among all full-survey respondents: Surveyed -> Aware -> Attempted -> Completed.
    Attempted and Completed are counted among the aware so that the funnel only narrows."""
    f = full(df)
    aware = f[f["aware_online"] == 1]
    return [("Full-survey respondents", len(f)), ("Aware online booking exists", len(aware)),
            ("Attempted a booking (aware)", int((aware["attempted_booking"] == 1).sum())),
            ("Completed a booking (aware)", int((aware["completed_booking"] == 1).sum()))]


def top_barrier(df):
    """Most-cited real barrier among those who never booked online (excludes 'no need' and vague answers)."""
    s, n = flag_share(df, "never__")
    skip = ("I have not needed", "No particular reason", "Another reason")
    s = s[[i for i in s.index if not i.startswith(skip)]]
    if s.empty:
        return "n/a", float("nan"), n
    return s.index[0], float(s.iloc[0]), n


def cx_table(df):
    names = {"cx_finding": "Finding the test", "cx_booking": "Completing the booking", "cx_payment": "Making payment",
             "cx_collection": "Sample collection", "cx_results": "Time to results"}
    rows = []
    for c, lab in names.items():
        s = pd.to_numeric(df[c], errors="coerce").dropna() if c in df else pd.Series(dtype=float)
        rows.append({"Touchpoint": lab, "n": len(s), "Satisfied %": float((s >= 4).mean() * 100) if len(s) else np.nan,
                     "Mean (1-5)": float(s.mean()) if len(s) else np.nan})
    return pd.DataFrame(rows)


def csat_proxy(df):
    cols = [c for c in ["cx_finding", "cx_booking", "cx_payment", "cx_collection", "cx_results"] if c in df]
    vals = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).ravel()
    vals = vals[~np.isnan(vals)]
    return (float((vals >= 4).mean() * 100) if len(vals) else np.nan), len(vals)


def intent_by_tier(df):
    f = full(df)
    g = f.groupby("offer_price").agg(n=("intent_top2", "count"), top2=("intent_top2", "mean"),
                                      definite=("intent_definite", "mean"), need=("need_flag", "mean"))
    g["top2_among_need"] = f[f["need_flag"] == 1].groupby("offer_price")["intent_top2"].mean()
    g["weighted_price"] = g["top2"] * g.index
    return g


def heat_matrix(df, row_col, row_order=None):
    """Would-book % (definitely + probably) by row group x price tier, with n and a small-n mask."""
    f = full(df).dropna(subset=[row_col, "offer_price", "intent_top2"])
    tiers = sorted(f["offer_price"].unique())
    rows = row_order or sorted(f[row_col].unique())
    rows = [r for r in rows if r in set(f[row_col])]
    z, txt = [], []
    for r in rows:
        zr, tr = [], []
        for t in tiers:
            cell = f[(f[row_col] == r) & (f["offer_price"] == t)]
            if len(cell) >= MIN_N:
                v = cell["intent_top2"].mean() * 100
                zr.append(v); tr.append(f"{v:.0f}%<br>n={len(cell)}")
            else:
                zr.append(None); tr.append(f"n={len(cell)}")
        z.append(zr); txt.append(tr)
    return rows, [tier_label(t) for t in tiers], z, txt


# ------------------------------------------------------------------ navigation
PAGES = [("app.py", "Home"), ("pages/1_Executive_Overview.py", "Executive Overview"),
         ("pages/2_Market_Digital_Funnel.py", "Market & Digital Funnel"),
         ("pages/3_Customer_Insights.py", "Customer Insights"),
         ("pages/4_Competitive_Barriers.py", "Competitive & Barriers"),
         ("pages/5_Strategic_Analytics.py", "Strategic Analytics")]


def nav_bar():
    cols = st.columns(len(PAGES))
    for col, (path, label) in zip(cols, PAGES):
        with col:
            st.page_link(path, label=label, use_container_width=True)


def nav_footer():
    """Navigation placed below the visualisations."""
    section_title("Dashboard modules")
    nav_bar()


# ------------------------------------------------------------------ UI components
def page_header(title, subtitle):
    st.markdown(f'<div class="page-header"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def metric_card(label, value, sub="", tone="good"):
    return (f'<div class="metric-card metric-{tone}"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>')


def show_cards(cards):
    cols = st.columns(len(cards))
    for col, html in zip(cols, cards):
        with col:
            st.markdown(html, unsafe_allow_html=True)


def insight_card(num, title, desc, tag=None):
    tag_html = f'<span class="insight-tag">{tag}</span>' if tag else ""
    return (f'<div class="insight-card"><div class="insight-header"><span class="insight-number">{num}</span>'
            f'<span class="insight-title">{title}</span></div><p class="insight-desc">{desc}</p>{tag_html}</div>')


def show_insights(cards, per_row=4):
    for i in range(0, len(cards), per_row):
        chunk = cards[i:i + per_row]
        cols = st.columns(per_row)
        for col, html in zip(cols, chunk):
            with col:
                st.markdown(html, unsafe_allow_html=True)
        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)


def caption(text):
    st.markdown(f'<div class="chart-caption">{text}</div>', unsafe_allow_html=True)


def style_fig(fig, height=340, legend=True, margin=None):
    fig.update_layout(height=height, margin=margin or dict(l=10, r=10, t=40, b=10), paper_bgcolor="white",
                      plot_bgcolor="white", font=dict(family="Arial, sans-serif", size=12, color="#334155"),
                      title_font=dict(size=14, color=MIDNIGHT), showlegend=legend,
                      legend=dict(orientation="h", y=-0.15, x=0))
    return fig


def empty_note(msg="Not enough respondents in this selection."):
    st.markdown(f'<div class="empty-note">{msg}</div>', unsafe_allow_html=True)


def require_data():
    df = load_data()
    if df.empty:
        st.error("Dataset not found. Run prepare_data.py to create data/synlab_dashboard.csv (see README).")
        st.stop()
    return df
