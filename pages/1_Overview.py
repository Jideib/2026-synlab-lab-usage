"""Overview: headline metrics, screening funnel, Lagos vs Abuja."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (
    BLUE, LIGHT, NAVY, banner, bootstrap, card, kpi, rate_ci, safe_col, sec,
    style,
)

d, c, F, FO, FA, base_note = bootstrap()

banner("Executive Overview",
       "Headline metrics, screening funnel and Lagos vs Abuja readiness")
st.caption(base_note)

r_aware = rate_ci(FO["aware_yes"])
r_booked = rate_ci(FO["booked_before"])
r_would = rate_ci(FO["would_book"])
r_need = rate_ci(FO["need_likely"])

k = st.columns(5)
kpi(k[0], f"{len(F):,}", "Eligible completes", f"{len(FO):,} online-ready")
kpi(k[1], f"{r_aware[0]:.0f}%", "Knew online booking exists",
    f"95% CI {r_aware[1]:.0f}–{r_aware[2]:.0f}%")
kpi(k[2], f"{r_booked[0]:.0f}%", "Booked online (12 months)",
    f"n={r_booked[3]:,}")
kpi(k[3], f"{r_would[0]:.0f}%", "Would book the package",
    f"95% CI {r_would[1]:.0f}–{r_would[2]:.0f}%")
kpi(k[4], f"{r_need[0]:.0f}%", "Likely to need it", "next 12 months")

sec("Screening funnel (fieldwork — not affected by filters)")
screens = [("Submissions in export", len(d))]
for col, label in [
    ("ok_consent", "Gave consent"),
    ("ok_s3a", "Live in Lagos or FCT Abuja"),
    ("ok_s4", "Help choose, arrange or pay for tests"),
    ("ok_s5", "Not lab industry, SYNLAB staff or organiser"),
    ("eligible", "Eligible after all screens"),
]:
    if col in d:
        screens.append(
            (label, int(pd.to_numeric(d[col], errors="coerce")
                        .fillna(0).eq(1).sum())))
screens.append(("Online-ready (analysis base)", int(d["online"].sum())))

fig = go.Figure(go.Bar(
    x=[n for _, n in screens][::-1],
    y=[l for l, _ in screens][::-1],
    orientation="h",
    marker_color=[NAVY] + [BLUE] * (len(screens) - 1),
    text=[f"{n:,}" for _, n in screens][::-1],
    textposition="outside", cliponaxis=False))
fig.update_xaxes(visible=False, range=[0, len(d) * 1.2])
st.plotly_chart(style(fig, 380), width="stretch")

sec("Lagos vs Abuja")
OA = FA[FA["online"]]
rows = {
    "Knew online booking exists": "aware_yes",
    "Booked online (12 months)": "booked_before",
    "Would book the package": "would_book",
}
fig = go.Figure()
for city, col in [("Lagos", NAVY), ("Abuja", LIGHT)]:
    s = OA[OA.city == city]
    vals = [rate_ci(s[f])[0] for f in rows.values()]
    ns = [rate_ci(s[f])[3] for f in rows.values()]
    fig.add_bar(name=f"{city} (n={ns[0]:,})", x=list(rows), y=vals,
                marker_color=col,
                text=[f"{v:.0f}%" for v in vals],
                textposition="outside")
fig.update_yaxes(visible=False, range=[0, 105])
fig.update_layout(barmode="group")
st.plotly_chart(style(fig, 380, True), width="stretch")

sec("Executive takeaways")
L, A = OA[OA.city == "Lagos"], OA[OA.city == "Abuja"]
t = st.columns(2)
with t[0]:
    card("Lagos is ready; Abuja is not yet",
         f"{rate_ci(L['would_book'])[0]:.0f}% of Lagos respondents would "
         f"book the package, vs {rate_ci(A['would_book'])[0]:.0f}% in Abuja. "
         f"Online booking history: {rate_ci(L['booked_before'])[0]:.0f}% vs "
         f"{rate_ci(A['booked_before'])[0]:.0f}%.", "01")
    card("Awareness is the gap",
         f"Only {rate_ci(OA['aware_yes'])[0]:.0f}% knew a lab test can be "
         f"booked online; the top reason for not booking is not knowing "
         f"it was possible.", "03")
with t[1]:
    top1 = safe_col(OA, c.top1)
    share = (top1 == "Confidence in accurate, reliable testing").mean() * 100
    card("Trust beats price",
         f"Accurate, reliable testing is the single most important factor for "
         f"{share:.0f}% of those who answered.", "02")
    card("Price matters in Abuja, not Lagos",
         "See Strategic Analytics for intent by price tier and city, with "
         "Wilson intervals.", "04")
