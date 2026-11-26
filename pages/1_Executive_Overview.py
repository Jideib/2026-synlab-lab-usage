# pages/1_Executive_Overview.py
import plotly.graph_objects as go
import streamlit as st

from utils import (CERULEAN, HALFBAKED, INTENT_COLORS, INTENT_ORDER, MIDNIGHT, BLUE_MED, caption, filter_bar, load_summary,
                   flag_share, fmt, full, funnel_counts, insight_card, intent_by_tier, metric_card, nav_footer,
                   page_header, pct_of, require_data, section_title, setup_page, show_cards, show_insights,
                   style_fig, tier_label, top_barrier, empty_note)

setup_page("Executive Overview")
raw = require_data()
page_header("Executive Overview", "Digital booking funnel, readiness and price response at a glance")
df = filter_bar(raw)
f = full(df)
if len(f) < 30:
    empty_note("Too few respondents match the current filters.")
    nav_footer()
    st.stop()

# ---------------------------------------------------------------- KPI rows
_total = load_summary().get("total_respondents")
aware = pct_of(f["aware_online"])
completed = pct_of(f["completed_booking"])
would_book = pct_of(f["intent_top2"])
digital = (df["digital_access"] == "I can do this myself").mean() * 100
home = (f["collect_pref"] == "At home").mean() * 100 if f["collect_pref"].notna().any() else float("nan")
ins_base = f["insurance"].dropna()
insured = (ins_base == "Insured").mean() * 100 if len(ins_base) else float("nan")
bar_label, bar_rate, bar_n = top_barrier(f)

section_title("1. Market volume and digital funnel")
show_cards([
    metric_card("Eligible respondents", f"{len(df):,}", f"{len(f):,} completed the full survey" + (f" &middot; {_total:,} surveyed in total" if _total else ""), "excellent"),
    metric_card("Online booking awareness", fmt(aware), f"{int(f['aware_online'].sum()):,} aware", "good"),
    metric_card("Online booking completion", fmt(completed), f"{int(f['completed_booking'].sum()):,} completed in 12 months", "average"),
    metric_card("Would book at tested price", fmt(would_book), "Definitely or probably", "attention"),
])
section_title("2. Readiness, preferences and barriers")
show_cards([
    metric_card("Digital capability", fmt(digital), "Can book or pay online themselves", "good"),
    metric_card("Home-collection preference", fmt(home), "Prefer home over a centre at equal price", "average"),
    metric_card("Health insurance / HMO", fmt(insured), "Have any HMO or insurance cover", "excellent"),
    metric_card("Top barrier", fmt(bar_rate), f"{bar_label[:44]}{'...' if len(bar_label) > 44 else ''}", "attention"),
])

# ---------------------------------------------------------------- donut + funnel
section_title("3. Who we surveyed and where the funnel narrows")
c1, c2 = st.columns([1, 1.4])
with c1:
    reg = df["region"].value_counts()
    fig = go.Figure(go.Pie(labels=reg.index, values=reg.values, hole=0.6, marker=dict(colors=[CERULEAN, MIDNIGHT, HALFBAKED]),
                           textinfo="label+percent", sort=False))
    style_fig(fig, 330, legend=False)
    fig.update_layout(title="Respondents by region", annotations=[dict(text=f"{len(df):,}", x=0.5, y=0.5, showarrow=False,
                                                                       font=dict(size=22, color=MIDNIGHT))])
    st.plotly_chart(fig, use_container_width=True)
with c2:
    steps = funnel_counts(df)
    fig = go.Figure(go.Funnel(y=[s[0] for s in steps], x=[s[1] for s in steps], textinfo="value+percent initial",
                              marker=dict(color=[MIDNIGHT, BLUE_MED, CERULEAN, HALFBAKED])))
    style_fig(fig, 330, legend=False)
    fig.update_layout(title="Booking funnel (aware respondents)")
    st.plotly_chart(fig, use_container_width=True)
caption("Attempted and completed are counted among respondents aware of online booking. "
        f"A further {int(((f['aware_online'] == 0) & (f['completed_booking'] == 1)).sum())} completed a booking without prior awareness "
        "(usually an HMO, employer or relative booked for them).")

# ---------------------------------------------------------------- price response distribution
section_title("4. Price response: how likely would you be to book at the price shown?")
tiers = sorted(f["offer_price"].dropna().unique())
fig = go.Figure()
for resp, col in zip(INTENT_ORDER, INTENT_COLORS):
    vals, labels = [], []
    for t in tiers:
        cell = f[f["offer_price"] == t]["intent"].dropna()
        vals.append((cell == resp).mean() * 100 if len(cell) else 0)
        labels.append(tier_label(t))
    fig.add_trace(go.Bar(y=labels, x=vals, name=resp.replace(" from the information shown", ""), orientation="h",
                         marker_color=col, text=[f"{v:.0f}%" if v >= 5 else "" for v in vals], textposition="inside"))
fig.update_layout(barmode="stack")
style_fig(fig, 280)
fig.update_layout(title="Stated booking likelihood by randomly assigned price (Basic Wellness Panel)", xaxis=dict(range=[0, 100], ticksuffix="%"))
st.plotly_chart(fig, use_container_width=True)
caption("Each respondent saw one randomly assigned price, so this is stated price response, not a measured willingness-to-pay.")

# ---------------------------------------------------------------- takeaway cards
tier = intent_by_tier(df)
lo, hi = tier.iloc[0], tier.iloc[-1]
no_know = pct_of(f[f["completed_booking"] == 0]["aware_online"].map({0.0: 1.0, 1.0: 0.0}))
section_title("5. Strategic takeaways")
show_insights([
    insight_card("01", "Close the awareness gap",
                 f"{fmt(aware)} are aware of online booking and {fmt(completed)} have completed one. "
                 f"Among people who have not booked, {fmt(no_know)} did not know it was possible.", "Awareness"),
    insight_card("02", "Digital readiness is high",
                 f"{fmt(digital)} manage online booking and payment themselves. The limiting factor is habit and awareness, not access.", "Readiness"),
    insight_card("03", "Intent is only mildly price-sensitive",
                 f"Would-book is {fmt(lo['top2'] * 100)} at {tier_label(tier.index[0])} and {fmt(hi['top2'] * 100)} at {tier_label(tier.index[-1])}. "
                 "Stated intent overstates real behaviour, so confirm with a pilot.", "Price response"),
    insight_card("04", "Trust and coverage shape the offer",
                 f"{fmt(home)} prefer home collection and {fmt(insured)} hold health cover. "
                 "Accuracy and a trusted lab remain the top reasons for choosing a lab.", "Offer design"),
], per_row=4)


nav_footer()
