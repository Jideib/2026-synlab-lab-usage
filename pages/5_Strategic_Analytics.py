# pages/5_Strategic_Analytics.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analytics as A
from utils import (load_data, CERULEAN, HALFBAKED, MIDNIGHT, MIN_N, BLUE_MED, caption, cat_share, empty_note, filter_bar, fmt, full,
                   heat_matrix, insight_card, intent_by_tier, nav_footer, page_header, pct_of, peso, require_data,
                   section_title, setup_page, show_cards, show_insights, style_fig, tier_label, metric_card, flag_share)

setup_page("Strategic Analytics")
raw = require_data()
page_header("Strategic Analytics & Price Response", "Stated price response, home-collection fees and a revenue scenario calculator")
df = filter_bar(raw)
f = full(df)
if len(f) < 30:
    empty_note("Too few respondents match the current filters.")
    nav_footer()
    st.stop()
st.markdown('<div class="chart-caption"><b>Read this page with care:</b> respondents saw one randomly assigned price for a '
            'Basic Wellness Panel. Results are <i>stated purchase intent at a given price</i>, not measured willingness to pay, '
            'and stated intent usually overstates what people actually do.</div>', unsafe_allow_html=True)

tier = intent_by_tier(df)

# ---------------------------------------------------------------- tier x region heatmap + bubble
c1, c2 = st.columns(2)
with c1:
    section_title("1. Would book by price tier and region")
    rows, tiers, z, txt = heat_matrix(df, "region")
    if not rows:
        empty_note()
    else:
        fig = go.Figure(go.Heatmap(z=z, x=tiers, y=rows, text=txt, texttemplate="%{text}", colorscale=[[0, "#EAF3F9"], [1, MIDNIGHT]],
                                   zmin=0, zmax=100, hoverongaps=False, showscale=False))
        style_fig(fig, 260, legend=False)
        fig.update_layout(title="Definitely or probably would book", xaxis=dict(side="top"), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        caption("Cells with fewer than 15 respondents are suppressed.")
with c2:
    section_title("2. Price vs stated booking likelihood")
    g = f.dropna(subset=["intent_top2", "offer_price"]).groupby(["region", "offer_price"]).agg(
        n=("intent_top2", "count"), top2=("intent_top2", "mean"), definite=("intent_definite", "mean")).reset_index()
    g = g[g["n"] >= MIN_N]
    if g.empty:
        empty_note()
    else:
        fig = go.Figure()
        for reg, colr in zip(sorted(g["region"].unique()), [CERULEAN, MIDNIGHT, HALFBAKED]):
            d = g[g["region"] == reg]
            fig.add_trace(go.Scatter(x=d["offer_price"], y=d["top2"] * 100, mode="markers+lines", name=f"{reg}: would book",
                                     marker=dict(size=[max(12, min(46, n / 6)) for n in d["n"]], color=colr, opacity=0.85), line=dict(color=colr, width=1.5),
                                     text=[f"n={n}" for n in d["n"]], hovertemplate="%{x:,.0f}: %{y:.0f}% (%{text})"))
        style_fig(fig, 300)
        fig.update_layout(title="Stated likelihood by price (bubble size = respondents)", xaxis=dict(title="Price shown (₦)", tickformat=",", tickvals=sorted(g["offer_price"].unique())),
                          yaxis=dict(title="Would book (%)", range=[0, 100]))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- home fee
section_title("3. Home-collection fee willingness")
h1, h2 = st.columns([1.3, 1])
with h1:
    amt = f["home_fee_ngn"].dropna()
    amt = amt[amt > 0]
    if len(amt) < 10:
        empty_note("Fewer than 10 respondents named a positive home-collection fee.")
    else:
        fig = go.Figure(go.Histogram(x=amt, nbinsx=16, marker_color=CERULEAN))
        style_fig(fig, 320, legend=False)
        fig.update_layout(title="Highest extra fee respondents would consider paying (positive amounts)", xaxis=dict(title="₦ per home collection", tickformat=","), yaxis=dict(title="Respondents"))
        st.plotly_chart(fig, use_container_width=True)
        caption(f"Base: {len(amt)} respondents. Median {peso(amt.median())}, mean {peso(amt.mean())}, 75th percentile {peso(amt.quantile(0.75))}.")
with h2:
    s = cat_share(f["home_fee_choice"]).sort_values()
    short = {"NGN 0, I would only consider it if there were no extra fee": "Only with no extra fee", "A positive amount: NGN________": "Would pay a positive amount",
             "I would not choose home collection even at no extra fee": "Would not choose home collection", "Do not know": "Do not know"}
    if s.empty:
        empty_note()
    else:
        fig = go.Figure(go.Bar(y=[short.get(i, i) for i in s.index], x=s.values, orientation="h", marker_color=BLUE_MED,
                               text=[f"{v:.0f}%" for v in s.values], textposition="outside", cliponaxis=False))
        style_fig(fig, 320, legend=False)
        fig.update_layout(title="Home-collection fee stance", xaxis=dict(visible=False), margin=dict(l=10, r=30, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- tier performance
section_title("4. Offer tier performance")
if tier.empty:
    empty_note()
else:
    perf = pd.DataFrame({"Price shown": [peso(p) for p in tier.index], "Respondents": tier["n"].astype(int).values,
                         "Would book %": (tier["top2"] * 100).round(1).values, "Definitely would %": (tier["definite"] * 100).round(1).values,
                         "Likely to need it %": (tier["need"] * 100).round(1).values,
                         "Would book, among those who need it %": (tier["top2_among_need"] * 100).round(1).values,
                         "Price x would book (₦)": tier["weighted_price"].round(0).astype(int).values})
    st.dataframe(perf, use_container_width=True, hide_index=True)
    caption("Price x would book is price multiplied by the share who would book: a rough index of intent-weighted revenue per respondent, not a forecast.")

# ---------------------------------------------------------------- logistic regression
REGIONS = tuple(sorted(df["region"].unique()))


@st.cache_data(show_spinner=False)
def fit_model(regions, outcome_key):
    data = load_data()
    sub = full(data[data["region"].isin(regions)])
    X, y = A.design_matrix(sub, outcome_key, include_price=(outcome_key == "intent_top2"))
    tab, stats = A.logit_fit(X, y)
    return tab, stats, X


section_title("5. What predicts online booking? Logistic regression")
st.markdown('<div class="chart-caption">A logistic regression estimates how each factor changes the <b>odds</b> of an outcome while holding the other factors '
            'constant. An odds ratio (OR) above 1 raises the odds, below 1 lowers them, and 1 means no effect. Bars show 95% confidence intervals; '
            'dark bars are statistically clear (p below 0.05). Reference groups: Abuja, age 25-34, uninsured, employed, books online alone, man.</div>',
            unsafe_allow_html=True)
outcome_label = st.selectbox("Outcome to explain", list(A.OUTCOMES), key="lr_outcome")
outcome_key = A.OUTCOMES[outcome_label]
try:
    tab, stats, X = fit_model(REGIONS, outcome_key)
    model_ok = True
except Exception as exc:  # too few cases, singular matrix, etc.
    model_ok = False
    empty_note(f"The model could not be estimated for this selection: {exc}")

if model_ok:
    show_cards([
        metric_card("Respondents in model", f"{stats['n']:,}", f"{stats['events']:,} in the outcome group ({stats['rate'] * 100:.0f}%)", "excellent"),
        metric_card("Discrimination (AUC)", f"{stats['auc']:.2f}", f"{A.auc_label(stats['auc']).capitalize()} separation of the two groups", "good"),
        metric_card("McFadden R²", f"{stats['r2']:.2f}", "Share of uncertainty explained", "average"),
        metric_card("Accuracy at 50% cut-off", f"{stats['accuracy'] * 100:.0f}%", f"vs {stats['baseline_accuracy'] * 100:.0f}% by always guessing the majority", "attention"),
    ])
    terms = tab[(tab["Term"] != "Intercept") & (~tab["Term"].str.contains("not stated"))].sort_values("OR")
    lr1, lr2 = st.columns([1.3, 1])
    with lr1:
        sig = (terms["p"] < 0.05) & (~terms["unstable"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=terms["OR"], y=terms["Term"], mode="markers",
                                 marker=dict(size=9, color=[MIDNIGHT if s else HALFBAKED for s in sig]),
                                 error_x=dict(type="data", symmetric=False, array=(terms["OR_high"] - terms["OR"]).tolist(),
                                              arrayminus=(terms["OR"] - terms["OR_low"]).tolist(), color="#94A3B8", thickness=1.5),
                                 hovertemplate="%{y}<br>OR %{x:.2f}<extra></extra>", showlegend=False))
        fig.add_vline(x=1, line_dash="dash", line_color="#64748B")
        style_fig(fig, max(380, 26 * len(terms) + 90), legend=False)
        fig.update_layout(title="Odds ratios with 95% confidence intervals", xaxis=dict(type="log", title="Odds ratio (log scale, 1 = no effect)"),
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with lr2:
        html = '<div class="swot-card swot-o" style="min-height:0;"><h4>What this model tells us</h4>'
        for head, items in A.executive_summary(X, tab, stats, outcome_key):
            html += (f'<div style="font-weight:700;color:#003765;font-size:13px;margin-top:10px;">{head}</div><ul>'
                     + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
        st.markdown(html + "</div>", unsafe_allow_html=True)
    show = terms.sort_values("OR", ascending=False)[["Term", "OR", "OR_low", "OR_high", "p"]].copy()
    show.columns = ["Factor", "Odds ratio", "CI low", "CI high", "p-value"]
    show[["Odds ratio", "CI low", "CI high"]] = show[["Odds ratio", "CI low", "CI high"]].round(2)
    show["p-value"] = show["p-value"].round(3)
    st.dataframe(show, use_container_width=True, hide_index=True)
    caption("Associations in a non-probability survey sample, not proof of cause. Placeholder groups for unanswered questions are estimated but not shown.")

# ---------------------------------------------------------------- k-means
@st.cache_data(show_spinner=False)
def run_segments(regions, k):
    data = load_data()
    sub = full(data[data["region"].isin(regions)])
    F = A.cluster_features(sub)
    Z = ((F - F.mean()) / F.std(ddof=0).replace(0, 1)).to_numpy()
    sil = {kk: A.silhouette(Z, A.kmeans(Z, kk)[0]) for kk in range(2, 7)}
    best = max(range(3, 6), key=lambda kk: sil[kk])
    kk = k or best
    labels, _, _ = A.kmeans(Z, kk)
    zdf = pd.DataFrame(Z, index=F.index, columns=F.columns)
    zprof = zdf.groupby(labels).mean()
    return {"F": F, "labels": pd.Series(labels, index=F.index), "sil": sil, "best": best, "k": kk, "zprof": zprof, "sub": sub.loc[F.index]}


def segment_action(z):
    if z["completed_booking"] > 0.5:
        return "Protect and grow: these people already book online. Keep the journey smooth and prompt repeat bookings."
    if z["digital_self"] < -0.8:
        return "Assisted route: older and less able to book alone. Use phone or WhatsApp help, doctor and pharmacist referral and printed results."
    if z["intent_top2"] > 0.5 and z["aware_online"] < 0:
        return "Convert: willing at the tested price but unaware of online booking. Awareness and one-step booking are the levers."
    if z["intent_top2"] < -0.5:
        return "Lower priority for the digital offer: low intent and little expected need. Revisit with value and trust messaging."
    return "Mixed profile: monitor and test messages before investing."


section_title("6. Who are our customers? K-means segmentation")
st.markdown('<div class="chart-caption">K-means groups respondents who look alike on ten characteristics (age, digital capability, insurance, awareness, '
            'online booking, stated intent, likely need, home-collection preference, test frequency, self-payment), each standardised so none dominates. '
            'The number of segments is yours to choose; the default is the best-separated option between 3 and 5.</div>', unsafe_allow_html=True)
seg0 = run_segments(REGIONS, 0)
k_pick = st.slider("Number of segments", 2, 6, int(seg0["best"]), key="km_k")
seg = run_segments(REGIONS, int(k_pick))
if len(seg["F"]) < 60:
    empty_note("Too few respondents for segmentation in this selection.")
else:
    sub, labels, zprof = seg["sub"], seg["labels"], seg["zprof"]
    names = {i: A.segment_name(zprof.loc[i]) for i in zprof.index}
    sil_k = seg["sil"][seg["k"]]
    cA, cB = st.columns([1, 1.6])
    with cA:
        ks = sorted(seg["sil"])
        fig = go.Figure(go.Bar(x=[str(kk) for kk in ks], y=[seg["sil"][kk] for kk in ks],
                               marker_color=[MIDNIGHT if kk == seg["k"] else HALFBAKED for kk in ks],
                               text=[f"{seg['sil'][kk]:.2f}" for kk in ks], textposition="outside", cliponaxis=False))
        style_fig(fig, 300, legend=False)
        fig.update_layout(title="Cluster separation (silhouette) by number of segments", xaxis=dict(title="Segments"),
                          yaxis=dict(visible=False, range=[0, max(seg["sil"].values()) * 1.3]))
        st.plotly_chart(fig, use_container_width=True)
        quality = "weak" if sil_k < 0.25 else "moderate" if sil_k < 0.5 else "strong"
        caption(f"Silhouette {sil_k:.2f}: <b>{quality}</b> structure. Values below 0.25 mean the segments are tendencies along a continuum, not sharply separate groups.")
    with cB:
        order = list(zprof.index)
        fig = go.Figure(go.Heatmap(z=zprof.loc[order].T.to_numpy(), x=[f"S{i + 1}" for i in order], y=[A.FEATURES[c] for c in zprof.columns],
                                   colorscale=[[0, HALFBAKED], [0.5, "#FFFFFF"], [1, MIDNIGHT]], zmid=0, zmin=-2, zmax=2,
                                   text=zprof.loc[order].T.round(1).to_numpy(), texttemplate="%{text}", showscale=False))
        style_fig(fig, 300, legend=False)
        fig.update_layout(title="Segment profile (standard deviations from the average respondent)", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    rows = []
    for i in zprof.index:
        g = sub[labels == i]
        rows.append({"Segment": f"S{i + 1}: {names[i]}", "Respondents": len(g), "Share %": round(len(g) / len(sub) * 100, 1),
                     "Aware %": round(pct_of(g["aware_online"]), 0), "Has booked online %": round(pct_of(g["completed_booking"]), 0),
                     "Would book %": round(pct_of(g["intent_top2"]), 0), "Insured %": round((g["insurance"] == "Insured").mean() * 100, 0),
                     "Lagos %": round((g["region"] == "Lagos").mean() * 100, 0)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    cards = []
    for n_, i in enumerate(zprof.index, 1):
        g = sub[labels == i]
        cards.append(insight_card(f"S{i + 1}", names[i],
                                  f"<b>{len(g) / len(sub) * 100:.0f}%</b> of respondents. {pct_of(g['completed_booking']):.0f}% have booked online and {pct_of(g['intent_top2']):.0f}% would book at the tested price. "
                                  f"<br><i>{segment_action(zprof.loc[i])}</i>", f"n={len(g)}"))
    show_insights(cards, per_row=min(4, len(cards)))
    caption("Segments describe the survey sample and are built only from the ten characteristics listed above.")

# ---------------------------------------------------------------- revenue calculator
section_title("7. Revenue scenario calculator")
need_default = pct_of(f["need_flag"])
need_default = int(round(need_default)) if not np.isnan(need_default) else 50
k1, k2, k3, k4 = st.columns(4)
with k1:
    prospects = st.number_input("Addressable prospects (people)", min_value=100, max_value=10_000_000, value=10_000, step=500, key="calc_pop")
with k2:
    need_pct = st.slider("Share who will need the package in 12 months (%)", 5, 100, min(max(need_default, 5), 100), key="calc_need")
with k3:
    realise = st.slider("Stated intent that turns into bookings (%)", 10, 100, 40, step=5, key="calc_real",
                        help="Survey intent overstates behaviour. This is an assumption, not a survey result.")
with k4:
    opts = [tier_label(p) for p in tier.index]
    pick = st.selectbox("Highlight price", opts, index=min(1, len(opts) - 1) if opts else 0, key="calc_tier") if opts else None

if tier.empty:
    empty_note()
else:
    rows = []
    for price, r in tier.iterrows():
        intent_need = r["top2_among_need"] if not np.isnan(r["top2_among_need"]) else r["top2"]
        bookings = prospects * need_pct / 100 * intent_need * realise / 100
        rows.append({"Price": tier_label(price), "price": price, "Expected bookings": bookings, "Revenue": bookings * price,
                     "Would book among need": intent_need * 100})
    sc = pd.DataFrame(rows)
    chosen = sc[sc["Price"] == pick].iloc[0]
    show_cards([
        metric_card("Expected bookings (12 months)", f"{chosen['Expected bookings']:,.0f}", f"at {pick}", "excellent"),
        metric_card("Revenue scenario", peso(chosen["Revenue"]), "Bookings x price", "good"),
        metric_card("Intent among those who need it", fmt(chosen["Would book among need"]), "From the survey, this price", "average"),
        metric_card("Realisation assumption", f"{realise}%", "Your assumption", "attention"),
    ])
    fig = go.Figure(go.Bar(x=sc["Price"], y=sc["Revenue"], marker_color=[MIDNIGHT if p == pick else HALFBAKED for p in sc["Price"]],
                           text=[peso(v) for v in sc["Revenue"]], textposition="outside", cliponaxis=False))
    style_fig(fig, 300, legend=False)
    fig.update_layout(title="Scenario revenue by price tier (same assumptions)", yaxis=dict(tickformat=",", title="₦"), margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    caption("Formula: prospects x share who need x would-book rate among those who need x realisation x price. "
            "A planning illustration built on stated intent; confirm with a pilot before committing budget.")

# ---------------------------------------------------------------- recommendations
section_title("8. Recommendations")
disc, _ = flag_share(f, "disc__")
doc = disc.get("A doctor or pharmacist", float("nan"))
aware = pct_of(f["aware_online"])
fee_amt = f["home_fee_ngn"].dropna()
fee_amt = fee_amt[fee_amt > 0]
lo, hi = (tier.iloc[0], tier.iloc[-1]) if len(tier) >= 2 else (None, None)
show_insights([
    insight_card("01", "Make online booking visible",
                 f"Only {fmt(aware)} know online booking exists. {fmt(doc)} want to hear from a doctor or pharmacist, so start with referral partners and in-lab prompts before paid media.", "Awareness"),
    insight_card("02", "Pilot more than one price",
                 (f"Would-book falls only from {fmt(lo['top2'] * 100)} to {fmt(hi['top2'] * 100)} across {tier_label(tier.index[0])} to {tier_label(tier.index[-1])}. "
                  "Run a live pilot at two or three prices before fixing the list price." if lo is not None else "Run a live price pilot before fixing the list price."), "Pricing"),
    insight_card("03", "Price home collection as an add-on",
                 (f"Among those willing to pay, the median extra fee is {peso(fee_amt.median())}. Many would only use home collection if it is free, so keep it optional." if len(fee_amt) >= 10
                  else "Keep home collection optional and test the fee."), "Collection"),
    insight_card("04", "Earn trust before asking for payment",
                 "Accuracy and a trusted lab drive choice. Show lab quality evidence, support WhatsApp booking and fix technical and payment friction.", "Trust"),
], per_row=4)


nav_footer()
