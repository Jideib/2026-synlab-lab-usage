# pages/3_Customer_Insights.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (AGE_ORDER, CERULEAN, HALFBAKED, MIDNIGHT, MIN_N, BLUE_MED, cat_share, caption, csat_proxy, empty_note,
                   filter_bar, fmt, full, heat_matrix, insight_card, intent_by_tier, nav_footer, page_header, pct_of,
                   require_data, section_title, setup_page, show_insights, style_fig, tier_label)

setup_page("Customer Insights")
raw = require_data()
page_header("Customer Insights & Segmentation", "Who books, who intends to book, and how segments differ")
df = filter_bar(raw, extended=True)
f = full(df)
if len(f) < 30:
    empty_note("Too few respondents match the current filters. Remove a filter to continue.")
    nav_footer()
    st.stop()
st.markdown(f'<div class="chart-caption">Selection: <b>{len(df):,}</b> respondents ({len(f):,} completed the full survey). '
            f'Cells and cards with fewer than {MIN_N} respondents are suppressed.</div>', unsafe_allow_html=True)


def heatmap(row_col, order, title):
    rows, tiers, z, txt = heat_matrix(df, row_col, order)
    if not rows:
        empty_note(); return
    fig = go.Figure(go.Heatmap(z=z, x=tiers, y=rows, text=txt, texttemplate="%{text}", colorscale=[[0, "#EAF3F9"], [1, MIDNIGHT]],
                               zmin=0, zmax=100, hoverongaps=False, showscale=False))
    style_fig(fig, max(260, 52 * len(rows) + 90), legend=False)
    fig.update_layout(title=title, yaxis=dict(autorange="reversed"), xaxis=dict(side="top"))
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------- heatmaps
section_title("1. Who would book at each price? (definitely or probably, by segment)")
h1, h2 = st.columns(2)
with h1:
    heatmap("age", AGE_ORDER, "Age group x price shown")
with h2:
    heatmap("employment", None, "Employment x price shown")
caption("Price tiers were randomly assigned. Each cell is the share answering definitely or probably would book.")

# ---------------------------------------------------------------- insurance vs behaviour
section_title("2. Insurance status and online booking behaviour")
ins = f.dropna(subset=["insurance"]).copy()
ins["status"] = np.where(ins["completed_booking"] == 1, "Completed a booking",
                  np.where(ins["started_not_completed"] == 1, "Started, not completed", "No online booking"))
b1, b2 = st.columns([1.3, 1])
with b1:
    order = [g for g in ["Insured", "Uninsured", "Not sure / no answer"] if (ins["insurance"] == g).sum() >= MIN_N]
    if not order:
        empty_note()
    else:
        fig = go.Figure()
        for stat, col in zip(["Completed a booking", "Started, not completed", "No online booking"], [MIDNIGHT, CERULEAN, "#DCE9F1"]):
            vals = [(ins[ins["insurance"] == g]["status"] == stat).mean() * 100 for g in order]
            fig.add_trace(go.Bar(y=[f"{g} (n={(ins['insurance'] == g).sum()})" for g in order], x=vals, name=stat, orientation="h",
                                 marker_color=col, text=[f"{v:.0f}%" if v >= 4 else "" for v in vals], textposition="inside"))
        fig.update_layout(barmode="stack")
        style_fig(fig, 300)
        fig.update_layout(title="Online booking behaviour by insurance status", xaxis=dict(range=[0, 100], ticksuffix="%"))
        st.plotly_chart(fig, use_container_width=True)
with b2:
    pay = f[[c for c in f.columns if c.startswith("paid_")]].apply(pd.to_numeric, errors="coerce")
    st.markdown("**Who paid for the last test**")
    names = {"paid_i_paid": "Respondent paid", "paid_an_hmo_or_insurer_paid": "HMO or insurer", "paid_the_person_having_the_test_o": "Patient or family",
             "paid_someone_else_paid": "Someone else", "paid_an_employer_paid_directly": "Employer"}
    base = pay.dropna(how="all")
    if len(base) >= MIN_N:
        s = (base[list(names)].mean() * 100).rename(index=names).sort_values()
        fig = go.Figure(go.Bar(y=s.index, x=s.values, orientation="h", marker_color=BLUE_MED, text=[f"{v:.0f}%" for v in s.values], textposition="outside", cliponaxis=False))
        style_fig(fig, 250, legend=False)
        fig.update_layout(xaxis=dict(visible=False), margin=dict(l=10, r=30, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        caption(f"Base: {len(base)} respondents who had a test.")
    else:
        empty_note()

# ---------------------------------------------------------------- collection & payment
section_title("3. Preferred collection and payment")
p1, p2 = st.columns(2)
for col, field, title in [(p1, "collect_pref", "Preferred sample collection (same price)"), (p2, "pay_pref", "Preferred payment arrangement")]:
    with col:
        s = cat_share(f[field]).sort_values()
        if s.empty:
            empty_note()
            continue
        short = {"A laboratory or collection centre": "Lab or collection centre", "At work, if available": "At work",
                 "I would not use an online booking service": "Would not use online booking",
                 "Pay electronically at collection or at the laboratory": "Pay electronically at the lab",
                 "Use HMO, insurer or employer billing": "HMO / employer billing",
                 "Pay cash at the laboratory, where accepted": "Cash at the lab"}
        fig = go.Figure(go.Bar(y=[short.get(i, i) for i in s.index], x=s.values, orientation="h", marker_color=CERULEAN,
                               text=[f"{v:.0f}%" for v in s.values], textposition="outside", cliponaxis=False))
        style_fig(fig, 320, legend=False)
        fig.update_layout(title=title, xaxis=dict(visible=False), margin=dict(l=10, r=30, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- area performance cards
section_title("4. Area performance")
level_label = st.radio("Area level", ["LGA / Area Council", "Neighbourhood"], horizontal=True, key="area_level")
by = "lga" if level_label.startswith("LGA") else "neighbourhood"
sizes = f.groupby(by).size()
keep = sizes[sizes >= MIN_N].sort_values(ascending=False).head(8).index.tolist()
if not keep:
    empty_note(f"No {level_label.lower()} has at least {MIN_N} full-survey respondents in this selection.")
else:
    cols = st.columns(4)
    for i, area in enumerate(keep):
        g = f[f[by] == area]
        csat, n_csat = csat_proxy(g)
        csat_txt = fmt(csat) if n_csat >= 20 else "n/a"
        html = (f'<div class="metric-card metric-small metric-good" style="margin-bottom:12px;"><div class="metric-label">{area}</div>'
                f'<div class="metric-value">{fmt(pct_of(g["aware_online"]))}</div>'
                f'<div class="metric-sub">aware &middot; {fmt(pct_of(g["completed_booking"]))} booked &middot; {csat_txt} satisfied<br>n={len(g)}</div></div>')
        with cols[i % 4]:
            st.markdown(html, unsafe_allow_html=True)
    caption("Big number = aware of online booking. Satisfied = share of CX ratings of 4 or 5 (shown only with at least 20 ratings).")

# ---------------------------------------------------------------- insights
section_title("5. Strategic insights for this selection")


def rate(frame, col):
    return pct_of(frame[col]) if len(frame) >= MIN_N else float("nan")


def ok(*xs):
    return all(not (isinstance(x, float) and np.isnan(x)) for x in xs)


u35, o45 = f[f["age"].isin(["18-24", "25-34"])], f[f["age"].isin(["45-54", "55-64", "65 or above"])]
ins_g, uni_g = f[f["insurance"] == "Insured"], f[f["insurance"] == "Uninsured"]
self_d = f[f["digital_access"] == "I can do this myself"]
help_d = f[f["digital_access"].isin(["I usually need someone to help me", "Someone else does it for me"])]
tier = intent_by_tier(df)
emp = f.dropna(subset=["employment", "intent_top2"]).groupby("employment").agg(n=("intent_top2", "count"), top2=("intent_top2", "mean"))
emp = emp[emp["n"] >= MIN_N].sort_values("top2", ascending=False)
reg = f.groupby("region").agg(n=("aware_online", "count"), aware=("aware_online", "mean"))
reg = reg[reg["n"] >= MIN_N]
fee = f["home_fee_choice"].dropna()
fee_zero = (fee.str.startswith("NGN 0")).mean() * 100 if len(fee) else float("nan")
paypref = cat_share(f["pay_pref"])
help_share = (f["digital_access"].notna() & (f["digital_access"] != "I can do this myself")).sum() / max(f["digital_access"].notna().sum(), 1) * 100
cp = f["collect_pref"].dropna()
home_pref = (cp == "At home").mean() * 100 if len(cp) else float("nan")
na = "Not enough respondents in this selection to compare."
cards = [
    insight_card("01", "Age and online booking",
                 (f"Under-35s completed an online booking at {fmt(rate(u35, 'completed_booking'))} against {fmt(rate(o45, 'completed_booking'))} for 45 and over."
                  if ok(rate(u35, "completed_booking"), rate(o45, "completed_booking")) else na), "Age"),
    insight_card("02", "Insurance and online booking",
                 (f"Insured respondents booked online at {fmt(rate(ins_g, 'completed_booking'))} against {fmt(rate(uni_g, 'completed_booking'))} for the uninsured."
                  if ok(rate(ins_g, "completed_booking"), rate(uni_g, "completed_booking")) else na), "Coverage"),
    insight_card("03", "Highest-intent employment group",
                 (f"{emp.index[0]} respondents show the highest would-book rate at {fmt(emp.iloc[0]['top2'] * 100)} (n={int(emp.iloc[0]['n'])})." if len(emp) else na), "Employment"),
    insight_card("04", "Home collection demand",
                 (f"{fmt(home_pref)} prefer home collection, and {fmt(fee_zero)} would only use it with no extra fee."
                  if f["collect_pref"].notna().any() else na), "Collection"),
    insight_card("05", "Payment preference",
                 (f"{paypref.index[0]} is preferred by {fmt(paypref.iloc[0])}." if len(paypref) else na), "Payment"),
    insight_card("06", "Help with online booking",
                 (f"{fmt(help_share)} rely on someone else or need help online. Their would-book rate is {fmt(rate(help_d, 'intent_top2'))} against {fmt(rate(self_d, 'intent_top2'))} for self-sufficient users."
                  if ok(rate(help_d, "intent_top2"), rate(self_d, "intent_top2")) else f"{fmt(help_share)} rely on someone else or need help to book online."), "Digital help"),
    insight_card("07", "Awareness by region",
                 (" and ".join(f"{r}: {fmt(row['aware'] * 100)} aware (n={int(row['n'])})" for r, row in reg.iterrows()) + "." if len(reg) else na), "Geography"),
    insight_card("08", "Price response",
                 (f"Would-book is {fmt(tier.iloc[0]['top2'] * 100)} at {tier_label(tier.index[0])} and {fmt(tier.iloc[-1]['top2'] * 100)} at {tier_label(tier.index[-1])}."
                  if len(tier) >= 2 else na), "Price"),
]
show_insights(cards, per_row=4)


nav_footer()
