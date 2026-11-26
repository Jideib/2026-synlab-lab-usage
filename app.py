# app.py - cover page
import streamlit as st

from utils import (caption, fmt, full, funnel_counts, load_data, load_summary, metric_card, nav_bar, pct_of,
                   render_hero_logo, require_data, setup_page, show_cards, show_insights, insight_card, section_title)

setup_page("Market Intelligence")
data = require_data()
f = full(data)

summary = load_summary()
eligible, n_full = len(data), len(f)
total = summary.get("total_respondents", eligible)   # all respondents after removing duplicates, before screening
aware = pct_of(f["aware_online"])
completed = pct_of(f["completed_booking"])
would_book = pct_of(f["intent_top2"])
n_regions = data["lga"].nunique()
lagos_pct = (data["region"] == "Lagos").mean() * 100
digital = (data["digital_access"] == "I can do this myself").mean() * 100
syn = f["synlab_used"].fillna("")
syn_used = syn.str.startswith("Yes").mean() * 100
syn_named = int((data["provider"] == "SYNLAB").sum())

st.markdown(
    '<div style="text-align:center; margin-top:4px;">' + render_hero_logo(70, "margin-bottom: 4px;") + "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="page-header" style="text-align:center; padding: 36px 40px;">'
    '<h1 style="font-size:38px; letter-spacing:-0.5px;">SYNLAB <span style="color:#7CB8D3;">Nigeria</span></h1>'
    '<p style="font-size:17px;">Laboratory Services Market Research and Digital Booking Intelligence</p>'
    '<div style="width:48px;height:3px;background:#7CB8D3;margin:16px auto 0;border-radius:2px;"></div>'
    '<div style="display:flex;justify-content:center;gap:48px;margin-top:22px;flex-wrap:wrap;">'
    '<div><span style="font-size:24px;font-weight:700;display:block;">Lagos &amp; FCT Abuja</span><span style="font-size:12px;opacity:.8;text-transform:uppercase;">Markets</span></div>'
    f'<div><span style="font-size:24px;font-weight:700;display:block;">{n_regions}</span><span style="font-size:12px;opacity:.8;text-transform:uppercase;">LGAs / Area Councils</span></div>'
    f'<div><span style="font-size:24px;font-weight:700;display:block;">{total:,}</span><span style="font-size:12px;opacity:.8;text-transform:uppercase;">Total respondents</span></div>'
    f'<div><span style="font-size:24px;font-weight:700;display:block;">{eligible:,}</span><span style="font-size:12px;opacity:.8;text-transform:uppercase;">Eligible</span></div>'
    '<div><span style="font-size:24px;font-weight:700;display:block;">Nov 2026</span><span style="font-size:12px;opacity:.8;text-transform:uppercase;">Fieldwork</span></div>'
    "</div></div>",
    unsafe_allow_html=True,
)

show_cards([
    metric_card("Total respondents", f"{total:,}", f"{eligible:,} eligible &middot; {n_full:,} completed the full survey", "excellent"),
    metric_card("Online booking awareness", fmt(aware), "Knew a test can be booked online", "good"),
    metric_card("Online booking completion", fmt(completed), "Completed one in the past 12 months", "average"),
    metric_card("Would book at tested price", fmt(would_book), "Definitely or probably (price test)", "attention"),
])

if "raw_submissions" in summary:
    caption(f"Respondent base: {summary['raw_submissions']:,} submissions received, {summary.get('duplicates_removed', 0):,} duplicate entries removed, "
            f"<b>{total:,} total respondents</b>, of whom <b>{eligible:,} were eligible</b> after screening and {n_full:,} completed the full questionnaire. "
            "All results below are based on the eligible respondents.")

section_title("Headline insights")
steps = funnel_counts(data)
show_insights([
    insight_card("01", "Awareness is the bottleneck",
                 f"Only <b>{fmt(aware)}</b> know a lab test can be arranged online, yet <b>{fmt(completed)}</b> have completed one. "
                 f"Of the {steps[1][1]} who are aware, {steps[3][1]} completed a booking.", "Digital funnel"),
    insight_card("02", "Most people can go online",
                 f"<b>{fmt(digital)}</b> can book or pay online themselves. Digital capability is not the main barrier; "
                 "knowing the option exists is.", "Readiness"),
    insight_card("03", "SYNLAB is known but rarely named",
                 f"<b>{fmt(syn_used)}</b> say they have used SYNLAB when asked directly, but only <b>{syn_named}</b> name it as "
                 "the lab used for their last test.", "Brand recall"),
], per_row=3)

section_title("Dashboard modules")
nav_bar()

st.markdown(
    '<div class="chart-caption" style="text-align:center; margin-top:24px;">'
    f"<b>SYNLAB Nigeria</b> &middot; Market Intelligence Platform &nbsp;|&nbsp; {total:,} respondents ({eligible:,} eligible) &middot; "
    "Lagos and FCT Abuja &middot; Stated intent from a survey, not observed behaviour</div>",
    unsafe_allow_html=True,
)
