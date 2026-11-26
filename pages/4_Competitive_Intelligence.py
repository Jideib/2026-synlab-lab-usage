"""Competitive Intelligence: online booking, barriers, satisfaction."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (
    BLUE, LIGHT, NAVY, banner, bootstrap, card, hbar, kpi, multi, pct,
    rate_ci, safe_col, sec, style, wrap,
)

d, c, F, FO, FA, base_note = bootstrap()

banner("Competitive Intelligence",
       "Online booking behaviour, barriers, satisfaction and channels")
st.caption(base_note)

done = safe_col(FO, c.done).fillna("")
k = st.columns(4)
kpi(k[0], f"{rate_ci(FO['aware_yes'])[0]:.0f}%", "Knew online booking exists", "")
kpi(k[1], f"{rate_ci(FO['booked_before'])[0]:.0f}%",
    "Completed an online booking", "past 12 months")
kpi(k[2], f"{done.str.contains('and paid online').mean() * 100:.0f}%",
    "Booked and paid online", "")
kpi(k[3], f"{done.str.contains('Started').mean() * 100:.0f}%",
    "Started, did not complete", "")

c1, c2 = st.columns(2)
with c1:
    sec("Online booking by city")
    g = (FO.groupby("city")
         .agg(aware=("aware_yes", "mean"),
              booked=("booked_before", "mean")) * 100)
    fig = go.Figure()
    fig.add_bar(name="Knew it was possible", x=g.index, y=g["aware"],
                marker_color=NAVY,
                text=[f"{v:.0f}%" for v in g["aware"]],
                textposition="outside")
    fig.add_bar(name="Completed a booking", x=g.index, y=g["booked"],
                marker_color=LIGHT,
                text=[f"{v:.0f}%" for v in g["booked"]],
                textposition="outside")
    fig.update_yaxes(visible=False, range=[0, 100])
    fig.update_layout(barmode="group")
    st.plotly_chart(style(fig, 340, True), width="stretch")
with c2:
    sec("Online booking by age group")
    g = FO.groupby(c.age)["booked_before"].mean().mul(100)
    fig = go.Figure(go.Bar(x=g.index, y=g.values, marker_color=BLUE,
                           text=[f"{v:.0f}%" for v in g.values],
                           textposition="outside"))
    fig.update_yaxes(visible=False, range=[0, 100])
    st.plotly_chart(style(fig, 340), width="stretch")

c1, c2 = st.columns(2)
with c1:
    sec("Where bookings happen (people who booked)")
    w = safe_col(FO, c.where).dropna()
    hbar(pct(w, 6), color=NAVY)
    st.caption(f"n={len(w)}")
with c2:
    sec("Why people have not booked online")
    nb_cols = [col for col in FO.columns
               if "main reasons you have not completed" in col.lower()
               and "/" in col]
    nb = FO[FO[nb_cols[0]].notna()] if nb_cols else FO.iloc[0:0]
    if nb_cols:
        hbar(multi(nb, "What are the main reasons you have not completed")
             .tail(8), color=BLUE)
        st.caption(f"n={len(nb)}")

c1, c2 = st.columns(2)
with c1:
    sec("Why bookings were abandoned")
    ab_cols = [col for col in FO.columns
               if "main reasons you did not complete" in col.lower()
               and "/" in col]
    ab = FO[FO[ab_cols[0]].notna()] if ab_cols else FO.iloc[0:0]
    if ab_cols:
        hbar(multi(ab, "What were the main reasons you did not complete")
             .tail(8), color=NAVY)
        st.caption(f"n={len(ab)}: small base, indicative only")
with c2:
    sec("Satisfaction at each step (online bookers)")
    steps = [
        ("Finding and understanding the test or package", "Finding"),
        ("Completing the booking", "Booking"),
        ("Making payment, if you paid", "Payment"),
        ("Sample collection, if completed", "Collection"),
        ("Time taken to receive results, if received", "Results"),
    ]
    rows = []
    for col, label in steps:
        if col in FO:
            s = FO[col].dropna()
            s = s[s != "Not applicable or not yet experienced"]
            rows.append((label,
                         s.isin(["Satisfied", "Very satisfied"]).mean() * 100,
                         len(s)))
    if rows:
        sat = pd.DataFrame(rows, columns=["step", "sat", "n"])
        fig = go.Figure(go.Bar(
            x=sat["sat"][::-1], y=[wrap(s, 30) for s in sat["step"][::-1]],
            orientation="h", marker_color=BLUE,
            text=[f"{v:.0f}% (n={n})"
                  for v, n in zip(sat["sat"][::-1], sat["n"][::-1])],
            textposition="outside", cliponaxis=False))
        fig.update_xaxes(visible=False, range=[0, 125])
        st.plotly_chart(style(fig, 300), width="stretch")

sec("Reading the competitive picture")
t = st.columns(3)
with t[0]:
    card("Reach is the problem, not the experience",
         "People who booked online are very satisfied at every step.", "01")
with t[1]:
    card("Not knowing is the top barrier",
         "Cost and security concerns each account for a small share of "
         "non-bookers.", "02")
with t[2]:
    card("Many routes to booking",
         "The lab's own site and WhatsApp lead, with HMO portals and "
         "third-party platforms behind.", "03")
