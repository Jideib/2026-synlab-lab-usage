"""Brand Health: SYNLAB use, drivers of choice, providers named."""
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from common import (
    BLUE, NAVY, SEQ, banner, bootstrap, card, hbar, kpi, multi, pct, safe_col,
    sec, style, wrap,
)

d, c, F, FO, FA, base_note = bootstrap()

banner("Brand Health and Awareness",
       "SYNLAB use, drivers of lab choice and providers named")
st.caption(base_note)

s = safe_col(FO, c.synlab).dropna()
if s.empty:
    st.info("SYNLAB usage question not found in this export.")
else:
    used = s.str.startswith("Yes")
    never = s.eq("No")
    unsure = s.str.contains("Not sure")
    k = st.columns(3)
    kpi(k[0], f"{used.mean() * 100:.0f}%", "Have used SYNLAB", "any time")
    kpi(k[1], f"{never.mean() * 100:.0f}%", "Never used SYNLAB", f"n={len(s)}")
    kpi(k[2], f"{unsure.mean() * 100:.0f}%", "Not sure", "")

    sec("Have you used SYNLAB? By city")
    ct = pd.crosstab(FO["city"], safe_col(FO, c.synlab), normalize="index") * 100
    if not ct.empty:
        fig = go.Figure()
        for i, col in enumerate(ct.columns):
            fig.add_bar(name=wrap(col, 30), y=ct.index, x=ct[col],
                        orientation="h", marker_color=SEQ[i % len(SEQ)],
                        text=[f"{v:.0f}%" for v in ct[col]],
                        textposition="inside")
        fig.update_layout(barmode="stack")
        fig.update_xaxes(visible=False)
        st.plotly_chart(style(fig, 250, True), width="stretch")

c1, c2 = st.columns(2)
with c1:
    sec("Top three things that matter when choosing a lab")
    hbar(multi(FO, "Which three things matter most"), color=NAVY)
with c2:
    sec("The single most important thing")
    hbar(pct(safe_col(FO, c.top1), 6), color=BLUE)

c1, c2 = st.columns(2)
with c1:
    sec("Information that would help people decide to book")
    hbar(pct(safe_col(FO, c.info), 6), color=BLUE)
with c2:
    sec("Providers named for the most recent test")
    pv = safe_col(F, c.provider).dropna().astype(str).str.lower().str.strip()

    def clean(x: str) -> str:
        for key, name in [("garki", "Garki Medical Laboratory"),
                          ("echo", "Echolab"), ("synlab", "SYNLAB")]:
            if key in x:
                return name
        return x.title()

    pv = pv.map(clean)
    top = pv.value_counts().head(8).sort_values()
    hbar(top, color=NAVY, fmt="{:.0f}")
    st.caption(f"{len(pv)} providers named; {pv.nunique()} distinct names "
               f"after light clean-up. Counts are mentions.")

sec("Brand health takeaways")
t = st.columns(3)
with t[0]:
    card("Unknown brand",
         "Three in four respondents have never used SYNLAB; competitors are "
         "fragmented, so no single online brand dominates.", "01")
with t[1]:
    card("Quality is the entry ticket",
         "Reliable, accurate testing and evidence of quality outrank price in "
         "what people say they need.", "02")
with t[2]:
    card("Lead with proof",
         "Accreditation, track record and named clinicians before promotions.",
         "03")
