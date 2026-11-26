"""Shared constants, styling, data loading, filters, helpers, models, and
the horizontal pill navigation used on every page.

The navigation is built with st.page_link() so it works inside Streamlit's
native pages/ multipage mode (the sidebar nav is hidden via CSS).
"""
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, silhouette_score
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "SYNLAB_survey_deduplicated_completes.csv"
LOGO = ROOT / "assets" / "synlab-logo.png"

NAVY, BLUE, LIGHT, ICE, MUTE = "#0B2A5B", "#1E88C9", "#7CC3F0", "#EAF3FA", "#4A5F7D"
SEQ = [NAVY, BLUE, LIGHT, "#9FB6D3", "#C9D8EA"]
SMALL_N = 30
SEED = 7

# --------------------------------------------------------------------------- #
# Page config & styling (call first on every page)
# --------------------------------------------------------------------------- #
def configure_page() -> None:
    st.set_page_config(
        page_title="SYNLAB Nigeria | Laboratory Services Survey",
        page_icon="🔬", layout="wide",
    )
    st.markdown(f"""<style>
    /* Hide Streamlit's auto sidebar page list (we render our own nav) */
    [data-testid="stSidebarNav"] {{ display: none; }}

    .block-container{{padding-top:1.2rem;max-width:1500px}}
    .banner{{background:linear-gradient(90deg,{NAVY},{BLUE});color:#fff;
            padding:26px 32px;border-radius:10px;margin-bottom:18px}}
    .banner h1{{color:#fff;font-size:30px;margin:0 0 4px 0}}
    .banner p{{color:#E3F0FA;margin:0;font-size:15px}}
    .kpi{{background:#fff;border-top:4px solid {BLUE};border-radius:8px;
         padding:16px 18px;text-align:center;height:100%}}
    .kpi .v{{font-size:30px;font-weight:700;color:{NAVY};line-height:1.1}}
    .kpi .l{{font-size:12px;letter-spacing:.08em;text-transform:uppercase;
            color:{MUTE};margin-top:6px}}
    .kpi .s{{font-size:12px;color:{MUTE};margin-top:4px}}
    .card{{background:#fff;border-left:5px solid {BLUE};border-radius:8px;
          padding:14px 18px;margin-bottom:10px}}
    .card b{{color:{NAVY}}} .card span{{color:{MUTE};font-size:14px}}
    .sec{{font-size:20px;font-weight:700;color:{NAVY};margin:26px 0 8px 0}}
    .note{{font-size:13px;color:{MUTE}}}
    .tag{{display:inline-block;background:{NAVY};color:#fff;border-radius:4px;
         padding:1px 8px;font-size:12px;margin-right:8px}}
    .small{{background:#FFF4E5;border-left:4px solid #E8A33D;
           padding:8px 12px;border-radius:6px;font-size:13px;
           color:#6B4A12;margin:6px 0}}

    /* --- Horizontal pill nav (uses st.page_link) ----------------------- */
    [data-testid="stPageLink-NavLink"] {{
        background:#fff; border:1px solid #CBD8EA; border-radius:8px;
        padding:12px 8px !important; text-align:center;
        color:{NAVY} !important; font-weight:500; font-size:14px;
        transition:all .15s ease; height:100%;
    }}
    [data-testid="stPageLink-NavLink"]:hover {{
        border-color:{BLUE}; color:{BLUE} !important;
    }}
    [data-testid="stPageLink-NavLink"][aria-current="page"] {{
        background:{ICE}; border-color:{BLUE}; color:{BLUE} !important;
        font-weight:700;
    }}
    [data-testid="stPageLink-NavLink"] p {{
        color: inherit !important; margin:0; font-size:14px;
    }}
    </style>""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Horizontal pill nav — used on every page
# --------------------------------------------------------------------------- #
PAGE_LINKS = [
    ("app.py",                              "Cover"),
    ("pages/1_Overview.py",                 "Overview"),
    ("pages/2_Brand_Health.py",             "Brand Health"),
    ("pages/3_Customer_Insights.py",        "Customer Insights"),
    ("pages/4_Competitive_Intelligence.py", "Competitive Intelligence"),
    ("pages/5_Strategic_Analytics.py",      "Strategic Analytics"),
]

def horizontal_nav() -> None:
    """Render the pill nav row at the top of every page."""
    cols = st.columns(len(PAGE_LINKS))
    for col, (page, label) in zip(cols, PAGE_LINKS):
        with col:
            st.page_link(page, label=label, use_container_width=True)

# --------------------------------------------------------------------------- #
# Column resolver
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

@dataclass
class Cols:
    raw: pd.DataFrame

    def find(self, *needles: str, multi: bool = False):
        n = [_norm(x) for x in needles]
        hits = [c for c in self.raw.columns if all(x in _norm(c) for x in n)]
        if not hits:
            return [] if multi else None
        return hits if multi else min(hits, key=len)

    @property
    def city(self):        return self.find("where", "currently", "live")
    @property
    def age(self):         return self.find("age", "group")
    @property
    def gender(self):      return self.find("gender")
    @property
    def emp(self):         return self.find("employment", "status")
    @property
    def ins(self):         return self.find("health", "insurance")
    @property
    def role(self):        return self.find("help choose", "arrange", "paid")
    @property
    def freq(self):        return self.find("past 12 months", "how many times")
    @property
    def reason(self):      return self.find("main reason", "test")
    @property
    def decided(self):     return self.find("who mainly decided")
    @property
    def paid(self):        return self.find("who paid", "test")
    @property
    def spend(self):       return self.find("approximately how much", "personally")
    @property
    def provider(self):    return self.find("which laboratory", "provider")
    @property
    def synlab(self):      return self.find("used", "synlab")
    @property
    def aware(self):       return self.find("know it was possible", "website", "whatsapp")
    @property
    def done(self):        return self.find("which of these", "personally done")
    @property
    def where(self):       return self.find("where did you complete", "online booking")
    @property
    def top1(self):        return self.find("which one", "matters most")
    @property
    def info(self):        return self.find("which one type of information")
    @property
    def intent(self):      return self.find("how likely", "book this package")
    @property
    def need(self):        return self.find("how likely", "actually need")
    @property
    def home(self):        return self.find("highest additional home", "fee")
    @property
    def amount(self):      return self.find("enter the amount")
    @property
    def collect(self):     return self.find("collection arrangement", "prefer")
    @property
    def pay(self):         return self.find("payment arrangement", "prefer")
    @property
    def results(self):     return self.find("how would you prefer", "results")
    @property
    def offer_price(self): return self.find("offer price") or self.find("total price")

# --------------------------------------------------------------------------- #
# Loading & derivation
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load() -> tuple[pd.DataFrame, Cols]:
    d = pd.read_csv(DATA, sep=";", low_memory=False)
    c = Cols(d)

    d["city"] = d[c.city].map({"FCT Abuja": "Abuja", "Lagos": "Lagos"})
    d["online"] = (pd.to_numeric(d["digital_ok"], errors="coerce")
                   .fillna(0).astype(int).eq(1))
    d["booked_before"] = d[c.done].fillna("").str.contains("Completed")
    d["aware_yes"] = d[c.aware].eq("Yes")
    d["would_book"] = d[c.intent].isin(["Definitely would", "Probably would"])
    d["judge"] = (d["online"] & d[c.intent].notna()
                  & d[c.intent].ne("Cannot judge from the information shown"))
    d["need_likely"] = d[c.need].isin(["Very likely", "Somewhat likely"])

    ins_raw = d[c.ins].fillna("")
    d["insured"] = np.select(
        [ins_raw.str.startswith("Yes"), ins_raw.eq("No")],
        ["Yes", "No"], default="Unknown")

    d["price_label"] = d[c.offer_price].map(
        {15000: "N15,000", 35000: "N35,000", 75000: "N75,000"})
    d["price_tier"] = d[c.offer_price].map(
        {15000: "p15", 35000: "p35", 75000: "p75"})
    return d, c

# --------------------------------------------------------------------------- #
# Sidebar filters
# --------------------------------------------------------------------------- #
def sidebar_filters(d: pd.DataFrame, c: Cols) -> tuple[list[str], list[str]]:
    with st.sidebar:
        if LOGO.exists():
            st.image(str(LOGO), width="stretch")
        st.markdown("**Filters**")
        cities = st.multiselect("City", ["Lagos", "Abuja"],
                                default=["Lagos", "Abuja"])
        ages_all = sorted(d[c.age].dropna().unique()) if c.age else []
        ages = st.multiselect("Age group", ages_all, default=ages_all)
        st.markdown(
            '<p class="note">Results are unweighted. Abuja is ~61% of the '
            'sample, so combined figures lean towards Abuja. Package intent is '
            'stated intent, not observed purchases.</p>',
            unsafe_allow_html=True)
    return cities, ages

# --------------------------------------------------------------------------- #
# Per-page bootstrap: config + nav + filters + sliced frames
# --------------------------------------------------------------------------- #
def bootstrap() -> tuple[pd.DataFrame, Cols, pd.DataFrame, pd.DataFrame,
                         pd.DataFrame, str]:
    """Call at the top of every page. Returns (d, c, F, FO, FA, base_note)."""
    configure_page()
    d, c = load()
    horizontal_nav()
    cities, ages = sidebar_filters(d, c)

    age_ok = d[c.age].isin(ages) if c.age else d.index == d.index
    city_ok = d["city"].isin(cities)
    FA = d[age_ok]
    F = d[age_ok & city_ok]
    FO = F[F["online"]]

    if F.empty:
        st.warning("No respondents match the selected filters.")
        st.stop()

    base_note = (
        f"Base: {len(F):,} eligible completes · "
        f"{len(FO):,} online-ready (answered the online-service block)."
    )
    return d, c, F, FO, FA, base_note

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def safe_col(df: pd.DataFrame, col: str | None) -> pd.Series:
    return df[col] if col and col in df else pd.Series(index=df.index, dtype=object)

def pct(s: pd.Series, top: int | None = None) -> pd.Series:
    v = s.dropna().value_counts(normalize=True).mul(100)
    return (v.head(top) if top else v).sort_values()

def rate_ci(s: pd.Series) -> tuple[float, float, float, int]:
    s = s.dropna()
    n = len(s)
    if n == 0:
        return (np.nan, np.nan, np.nan, 0)
    k = int(s.astype(bool).sum())
    lo, hi = proportion_confint(k, n, method="wilson")
    return (k / n * 100, lo * 100, hi * 100, n)

def multi(df: pd.DataFrame, prefix: str) -> pd.Series:
    pref = _norm(prefix)
    cols = [c for c in df.columns if _norm(c).startswith(pref) and "/" in c]
    if not cols:
        return pd.Series(dtype=float)
    s = pd.Series({
        c.split("/")[-1]: pd.to_numeric(df[c], errors="coerce").mean() * 100
        for c in cols
    }).dropna()
    return s.sort_values()

def wrap(label, n: int = 42) -> str:
    return "<br>".join(textwrap.wrap(str(label), n)) or str(label)

def style(fig: go.Figure, h: int = 340, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=h, margin=dict(l=10, r=30, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=NAVY, size=13), showlegend=legend,
        legend=dict(orientation="h", y=1.12))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig

def hbar(s: pd.Series, h: int | None = None, color: str = BLUE,
         fmt: str = "{:.0f}%") -> None:
    if s.empty:
        st.info("No data for this question in the current filter.")
        return
    s = s.sort_values()
    fig = go.Figure(go.Bar(
        x=s.values, y=[wrap(i) for i in s.index], orientation="h",
        marker_color=color,
        text=[fmt.format(v) for v in s.values], textposition="outside",
        cliponaxis=False))
    fig.update_xaxes(visible=False, range=[0, max(s.max() * 1.18, 1)])
    st.plotly_chart(style(fig, h or max(200, 46 * len(s) + 40)),
                    width="stretch")

def kpi(col, value: str, label: str, sub: str = "") -> None:
    col.markdown(
        f'<div class="kpi"><div class="v">{value}</div>'
        f'<div class="l">{label}</div><div class="s">{sub}</div></div>',
        unsafe_allow_html=True)

def banner(title: str, sub: str) -> None:
    st.markdown(
        f'<div class="banner"><h1>{title}</h1><p>{sub}</p></div>',
        unsafe_allow_html=True)

def sec(t: str) -> None:
    st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)

def card(title: str, body: str, tag: str = "") -> None:
    t = f'<span class="tag">{tag}</span>' if tag else ""
    st.markdown(
        f'<div class="card">{t}<b>{title}</b><br><span>{body}</span></div>',
        unsafe_allow_html=True)

def small_n_note(n: int) -> None:
    if 0 < n < SMALL_N:
        st.markdown(
            f'<div class="small">⚠ Small base (n={n}). Indicative only.</div>',
            unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Cached models (all-respondent, ignore filters by design)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Fitting models…")
def fit_models(_d: pd.DataFrame, _c: Cols) -> dict:
    m = _d[_d["judge"]].copy()
    m["y"] = m["would_book"].astype(int)
    m["ageg"] = safe_col(m, _c.age).replace({"55-64": "55+", "65 or above": "55+"})
    m["aware_i"] = m["aware_yes"].astype(int)
    m["bb"] = m["booked_before"].astype(int)
    m["syn"] = safe_col(m, _c.synlab).fillna("").str.startswith("Yes").astype(int)
    m["pr"] = m["price_tier"]
    m["city_"] = m["city"]

    formula = ('y ~ C(pr, Treatment("p15")) + C(city_, Treatment("Abuja")) '
               '+ C(ageg, Treatment("25-34")) + C(insured, Treatment("No")) '
               '+ aware_i + bb + syn')
    try:
        res = smf.logit(formula, m).fit(disp=0)
    except Exception as e:
        return {"error": f"Model failed: {e}"}

    me = res.get_margeff().summary_frame()
    labels = {
        "p35": "Price N35,000 (vs N15,000)",
        "p75": "Price N75,000 (vs N15,000)",
        "T.Lagos": "Lagos (vs Abuja)",
        "18-24": "Aged 18-24 (vs 25-34)",
        "35-44": "Aged 35-44 (vs 25-34)",
        "45-54": "Aged 45-54 (vs 25-34)",
        "55+":   "Aged 55+ (vs 25-34)",
        "T.Yes": "Insured (vs uninsured)",
        "T.Unknown": "Insurance unknown",
        "aware_i": "Knew online booking existed",
        "bb":     "Has booked online before",
        "syn":    "Has used SYNLAB",
    }
    rows = []
    for k, v in me.iterrows():
        name = next((lab for t, lab in labels.items() if t in k), k)
        rows.append((name, v.iloc[0] * 100, v.iloc[2] * 100,
                     v.iloc[3] * 100, v.iloc[4]))
    eff = pd.DataFrame(rows, columns=["Factor", "pp", "lo", "hi", "p"])

    el = []
    for city in ["Abuja", "Lagos"]:
        for pr, lab in [("p15", "N15,000"), ("p35", "N35,000"), ("p75", "N75,000")]:
            s = m[(m.city_ == city) & (m.pr == pr)]["y"]
            r, lo, hi, n = rate_ci(s.astype(bool))
            el.append((city, lab, r, lo, hi, n))
    el = pd.DataFrame(el, columns=["city", "price", "rate", "lo", "hi", "n"])

    fit = dict(n=len(m), r2=res.prsquared,
               auc=roc_auc_score(m["y"], res.predict(m)))
    return {"eff": eff, "fit": fit, "price": el, "model": m}

@st.cache_data(show_spinner="Segmenting…")
def segment(_d: pd.DataFrame, _c: Cols) -> dict:
    m = _d[_d["judge"]].copy()

    def simp(s, keep):
        s = safe_col(m, s)
        return s.where(s.isin(keep), "Other")

    X = pd.DataFrame({
        "collect": simp(_c.collect,
                        ["A laboratory or collection centre", "At home"]),
        "pay": simp(_c.pay, [
            "Pay electronically at collection or at the laboratory",
            "Pay online before collection",
            "Use HMO, insurer or employer billing",
            "Pay cash at the laboratory, where accepted"]),
        "results": simp(_c.results, [
            "A printed copy collected from the laboratory",
            "A secure patient portal or app",
            "A secure link sent by WhatsApp",
            "A secure link sent by email"]),
        "top": simp(_c.top1, ["Confidence in accurate, reliable testing"]),
        "aware": m["aware_yes"].astype(str),
        "booked": m["booked_before"].astype(str),
    })
    Z = pd.get_dummies(X, drop_first=False).astype(float)
    if Z.shape[0] < 4 or Z.shape[1] < 2:
        return {"error": "Too little data to segment."}

    k = min(4, max(2, Z.shape[0] // 50))
    km = KMeans(k, n_init=50, random_state=SEED).fit(Z)
    labels = pd.Series(km.labels_, index=m.index)
    sil = silhouette_score(Z, km.labels_)

    names = {}
    means = pd.DataFrame({
        "booked": m["booked_before"].astype(int),
        "quality": (safe_col(m, _c.top1)
                    == "Confidence in accurate, reliable testing").astype(int),
        "lab": (safe_col(m, _c.collect)
                == "A laboratory or collection centre").astype(int),
    }).groupby(labels).mean()

    remaining = set(means.index)
    for label, criterion in [
        ("Digital-ready", "booked"),
        ("Quality-first, in-person", "quality"),
        ("In-person, needs guidance", "lab"),
    ]:
        if not remaining:
            break
        s = means.loc[list(remaining), criterion].idxmax()
        names[s] = label
        remaining.discard(s)
    for s in remaining:
        names[s] = "Flexible, not yet engaged"

    m["segment"] = labels.map(names)
    xy = PCA(2, random_state=SEED).fit_transform(Z)
    m["pc1"], m["pc2"] = xy[:, 0], xy[:, 1]

    seg = m.groupby("segment").agg(
        n=("would_book", "size"),
        share=("would_book", lambda s: len(s) / len(m) * 100),
        lagos=("city", lambda s: (s == "Lagos").mean() * 100),
        booked=("booked_before", lambda s: s.mean() * 100),
        would=("would_book", lambda s: s.mean() * 100),
    ).reset_index()
    return {"seg": seg, "points": m[["segment", "pc1", "pc2"]], "sil": sil}
