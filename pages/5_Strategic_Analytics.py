"""Strategic Analytics: price response, intent model, segments, roadmap."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (
    BLUE, LIGHT, MUTE, NAVY, SEQ, banner, bootstrap, card, fit_models, kpi,
    sec, segment, style,
)

d, c, F, FO, FA, base_note = bootstrap()

banner("Strategic Analytics and Advanced Models",
       "Price response, booking-intent model, customer segments and priorities")
st.caption("Models use all respondents who could judge the package, whatever "
           "the sidebar filters, so results match the report.")

models = fit_models(d, c)
if "error" in models:
    st.error(models["error"])
    st.stop()

eff, fit, el = models["eff"], models["fit"], models["price"]

k = st.columns(4)
kpi(k[0], f"{fit['n']}", "Respondents in models", "could judge the package")
kpi(k[1], f"{fit['r2']:.2f}", "Pseudo R-squared", "modest fit")
kpi(k[2], f"{fit['auc']:.2f}", "AUC (same data)", "ranks well, not validated")
seg = segment(d, c)
sil = seg.get("sil", 0) if "error" not in seg else 0
kpi(k[3], f"{sil:.2f}", "Segment silhouette", "weak separation")

sec("Price response by city (would probably or definitely book)")
fig = go.Figure()
for city, col in [("Abuja", LIGHT), ("Lagos", NAVY)]:
    s = el[el.city == city]
    fig.add_trace(go.Scatter(
        x=s["price"], y=s["rate"],
        mode="lines+markers+text", name=city,
        line=dict(color=col, width=3),
        text=[f"{v:.0f}%" for v in s["rate"]], textposition="top center",
        error_y=dict(type="data", symmetric=False,
                     array=s["hi"] - s["rate"],
                     arrayminus=s["rate"] - s["lo"], color=col)))
fig.update_yaxes(range=[0, 105], title="% would book (95% interval)",
                 showgrid=True, gridcolor="#D5E4F3")
st.plotly_chart(style(fig, 380, True), width="stretch")
st.caption("Price tiers were randomly assigned. Based on respondents who "
           "could judge the package, so figures are a little higher than the "
           "all-respondent figures on the Overview.")

sec("What drives booking intent (logistic regression, effect in points)")
e = eff.sort_values("pp")
fig = go.Figure(go.Scatter(
    x=e["pp"], y=e["Factor"], mode="markers",
    marker=dict(size=11,
                color=[NAVY if p < 0.05 else "#9FB6D3" for p in e["p"]]),
    error_x=dict(type="data", symmetric=False,
                 array=e["hi"] - e["pp"], arrayminus=e["pp"] - e["lo"],
                 color="#9FB6D3")))
fig.add_vline(x=0, line_color=MUTE)
fig.update_xaxes(title="Change in probability of booking (points), "
                       "all else equal",
                 showgrid=True, gridcolor="#D5E4F3")
st.plotly_chart(style(fig, 460), width="stretch")
st.caption("Dark points are statistically reliable (p<0.05); light points "
           "are not. Insurance looks important in raw figures but is mostly "
           "a city effect.")

if "error" not in seg:
    sec("Customer segments (k-means)")
    sg_df, pcs, sil = seg["seg"], seg["points"], seg["sil"]
    order = ["Digital-ready", "In-person, needs guidance",
             "Quality-first, in-person", "Flexible, not yet engaged"]
    c1, c2 = st.columns([1, 1])
    with c1:
        sg = sg_df.set_index("segment").reindex(order).dropna(how="all")
        show = pd.DataFrame({
            "Share": sg["share"].map("{:.0f}%".format),
            "In Lagos": sg["lagos"].map("{:.0f}%".format),
            "Booked online": sg["booked"].map("{:.0f}%".format),
            "Would book": sg["would"].map("{:.0f}%".format),
        })
        st.dataframe(show, width="stretch")
        st.caption(f"Silhouette {sil:.2f}: the groups overlap. Treat them as "
                   f"working personas, not hard boundaries.")
    with c2:
        fig = go.Figure()
        for i, nm in enumerate(order):
            s = pcs[pcs.segment == nm]
            fig.add_trace(go.Scatter(
                x=s["pc1"], y=s["pc2"], mode="markers", name=nm,
                marker=dict(size=7, color=SEQ[i], opacity=0.7)))
        fig.update_xaxes(title="Preference dimension 1", showgrid=False)
        fig.update_yaxes(title="Preference dimension 2", showgrid=False)
        st.plotly_chart(style(fig, 320, True), width="stretch")

sec("What we did not model")
st.markdown(
    "Marketing mix modelling needs spend and sales over time; a single "
    "survey does not contain them. Channel questions here show where people "
    "want to hear about labs, not what a campaign returned.")

sec("Prioritised action roadmap (for discussion)")
for tag, title, body in [
    ("P1 HIGH", "Lead online booking in Lagos",
     "Offer the package at mid or top tier; demand is high and does not fall "
     "with price."),
    ("P1 HIGH", "Build awareness and quality proof",
     "Accreditation, track record and named referring clinicians before "
     "price promotions."),
    ("P2 MEDIUM", "Win referrals from doctors and pharmacists",
     "The preferred source for hearing about lab services."),
    ("P2 MEDIUM", "Test the lower price in Abuja",
     "Keep in-person collection and printed results as the default there."),
    ("P3 LOW", "Home collection as an add-on",
     "Half would pay nothing extra; those willing to pay offer a median of "
     "N3,000."),
]:
    card(title, body, tag)
