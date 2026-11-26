"""Customer Insights: who arranges tests, why, who pays, preferences."""
import plotly.graph_objects as go
import streamlit as st

from common import (
    BLUE, LIGHT, NAVY, banner, bootstrap, card, hbar, multi, pct, sec, style,
)

d, c, F, FO, FA, base_note = bootstrap()

banner("Customer Insights and Segmentation",
       "Who arranges tests, why, who pays, and service preferences")
st.caption(f"Profile questions: {len(F):,} eligible completes · "
           f"preferences: {len(FO):,} online-ready.")

c1, c2, c3 = st.columns(3)
with c1:
    sec("Age group")
    a = F[c.age].value_counts(normalize=True).mul(100).sort_index()
    fig = go.Figure(go.Bar(x=a.index, y=a.values, marker_color=BLUE,
                           text=[f"{v:.0f}%" for v in a.values],
                           textposition="outside"))
    fig.update_yaxes(visible=False, range=[0, a.max() * 1.2])
    st.plotly_chart(style(fig, 300), width="stretch")
with c2:
    sec("Gender")
    hbar(pct(F[c.gender]), h=240, color=NAVY)
with c3:
    sec("Employment")
    hbar(pct(F[c.emp], 5), h=300, color=BLUE)

c1, c2, c3 = st.columns(3)
with c1:
    sec("Insurance by city")
    ct = (F[F.insured != "Unknown"].groupby("city")["insured"]
          .apply(lambda s: (s == "Yes").mean() * 100))
    fig = go.Figure(go.Bar(
        x=ct.index, y=ct.values,
        marker_color=[NAVY, LIGHT][: len(ct)],
        text=[f"{v:.0f}%" for v in ct.values], textposition="outside"))
    fig.update_yaxes(visible=False, range=[0, 100])
    st.plotly_chart(style(fig, 300), width="stretch")
with c2:
    sec("Who arranges tests")
    hbar(pct(F[c.role]), h=240, color=BLUE)
with c3:
    sec("Tests in past 12 months")
    hbar(pct(F[c.freq]), h=280, color=NAVY)

c1, c2 = st.columns(2)
with c1:
    sec("Main reason for the most recent test")
    hbar(pct(F[c.reason], 6), color=NAVY)
with c2:
    sec("Who chose the lab")
    hbar(pct(F[c.decided], 6), color=BLUE)

c1, c2 = st.columns(2)
with c1:
    sec("Who paid")
    hbar(pct(F[c.paid], 6), color=BLUE)
with c2:
    sec("What respondents paid personally")
    hbar(pct(F[c.spend], 7), color=NAVY)

sec("Service preferences (online-ready respondents)")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Collection**")
    hbar(pct(FO[c.collect]), h=260, color=BLUE)
with c2:
    st.markdown("**Payment**")
    hbar(pct(FO[c.pay], 5), h=300, color=NAVY)
with c3:
    st.markdown("**Results delivery**")
    hbar(pct(FO[c.results], 5), h=300, color=BLUE)

c1, c2 = st.columns(2)
with c1:
    sec("Where people prefer to hear about lab services")
    hbar(multi(FO, "Where would you prefer to hear").tail(7), color=NAVY)
with c2:
    sec("Services expected in the next 12 months")
    hbar(multi(FO, "For yourself or someone you help"), color=BLUE)
