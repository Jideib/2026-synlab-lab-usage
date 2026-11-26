# pages/2_Market_Digital_Funnel.py
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (BLUE_MED, CERULEAN, HALFBAKED, MIDNIGHT, MIN_N, caption, csat_proxy, cx_table, empty_note,
                   filter_bar, flag_share, fmt, full, funnel_counts, insight_card, nav_footer, page_header, pct_of, require_data,
                   section_title, setup_page, show_insights, style_fig)

setup_page("Market & Digital Funnel")
raw = require_data()
page_header("Market & Digital Funnel", "Where awareness, attempts and completed bookings drop off, and what people say about it")
df = filter_bar(raw)
f = full(df)
if len(f) < 30:
    empty_note("Too few respondents match the current filters.")
    nav_footer()
    st.stop()


def hbar(series, title, color=CERULEAN, height=360, pct=True, top=None):
    s = series.head(top) if top else series
    s = s.iloc[::-1]
    fig = go.Figure(go.Bar(y=[str(i) for i in s.index], x=s.values, orientation="h", marker_color=color,
                           text=[f"{v:.0f}%" for v in s.values] if pct else s.values, textposition="outside", cliponaxis=False))
    style_fig(fig, height, legend=False)
    fig.update_layout(title=title, xaxis=dict(visible=False), margin=dict(l=10, r=30, t=40, b=10))
    return fig


# ---------------------------------------------------------------- awareness by LGA + funnel
c1, c2 = st.columns(2)
with c1:
    section_title("1. Awareness of online booking by LGA / Area Council")
    g = f.groupby("lga").agg(n=("aware_online", "count"), aware=("aware_online", "mean"))
    g = g[g["n"] >= MIN_N].sort_values("aware", ascending=False)
    if g.empty:
        empty_note()
    else:
        s = g["aware"] * 100
        s.index = [f"{i} (n={int(g.loc[i, 'n'])})" for i in s.index]
        st.plotly_chart(hbar(s, "Aware that a test can be booked online", height=max(300, 26 * len(s) + 80)), use_container_width=True)
        caption(f"Areas with fewer than {MIN_N} respondents are hidden.")
with c2:
    section_title("2. Funnel drop-off")
    steps = funnel_counts(df)
    fig = go.Figure(go.Funnel(y=[s[0] for s in steps], x=[s[1] for s in steps], textinfo="value+percent previous",
                              marker=dict(color=[MIDNIGHT, BLUE_MED, CERULEAN, HALFBAKED])))
    style_fig(fig, 300, legend=False)
    st.plotly_chart(fig, use_container_width=True)
    tab = pd.DataFrame({"Stage": [s[0] for s in steps], "Respondents": [s[1] for s in steps]})
    tab["% of previous stage"] = [100.0] + [round(steps[i][1] / steps[i - 1][1] * 100, 1) if steps[i - 1][1] else 0 for i in range(1, len(steps))]
    tab["% of aware"] = [None] + [round(s[1] / steps[1][1] * 100, 1) if steps[1][1] else 0 for s in steps[1:]]
    st.dataframe(tab, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- reasons + channels
c3, c4 = st.columns(2)
with c3:
    section_title("3. Why bookings were not completed")
    s, n = flag_share(f[(f["started_not_completed"] == 1) & (f["completed_booking"] != 1)], "abandon__")
    if n < 10:
        empty_note(f"Only {n} respondents started but did not complete a booking in this selection.")
    else:
        st.plotly_chart(hbar(s, "Reasons for not completing a started booking", MIDNIGHT, 360), use_container_width=True)
        caption(f"Base: {n} respondents who started an online booking and did not complete it. Multiple answers allowed. Small base: read as directional.")
with c4:
    section_title("4. Where people want to hear about lab services")
    s, n = flag_share(f, "disc__")
    s = s[[i for i in s.index if i != "Another channel"]]
    st.plotly_chart(hbar(s, "Preferred discovery channels", CERULEAN, 360), use_container_width=True)
    caption(f"Base: {n} respondents. Multiple answers allowed.")

# ---------------------------------------------------------------- drill-down + radar
section_title("5. Location drill-down")
d1, d2, d3 = st.columns([1, 1.4, 2.6])
with d1:
    reg_opts = ["All regions"] + sorted(df["region"].dropna().unique().tolist())
    sel_reg = st.selectbox("Region", reg_opts, key="dd_region")
sub = df if sel_reg == "All regions" else df[df["region"] == sel_reg]
with d2:
    lga_opts = ["All LGAs / Area Councils"] + sorted(sub["lga"].dropna().unique().tolist())
    sel_lga = st.selectbox("LGA / Area Council", lga_opts, key="dd_lga")
sub = sub if sel_lga.startswith("All") else sub[sub["lga"] == sel_lga]
level = "lga" if sel_lga.startswith("All") else "neighbourhood"


def summarise(frame, by):
    rows = []
    for k, grp in frame.groupby(by):
        fg = full(grp)
        csat, n_csat = csat_proxy(fg)
        rows.append({"Area": k, "Respondents": len(grp), "Aware %": pct_of(fg["aware_online"]),
                     "Completed booking %": pct_of(fg["completed_booking"]), "Would book %": pct_of(fg["intent_top2"]),
                     "Satisfied % (CX)": csat if n_csat >= 10 else float("nan")})
    out = pd.DataFrame(rows)
    return out.sort_values("Respondents", ascending=False) if not out.empty else out


tab = summarise(sub, level)
if tab.empty:
    empty_note()
else:
    if level == "neighbourhood":
        small = tab[tab["Respondents"] < 3]
        tab = tab[tab["Respondents"] >= 3]
        caption(f"Showing neighbourhoods with at least 3 respondents ({len(small)} smaller neighbourhoods not shown).")
    tab = tab.round(1)
    tab.columns = [("Neighbourhood" if (c == "Area" and level == "neighbourhood") else ("LGA / Area Council" if c == "Area" else c)) for c in tab.columns]
    st.dataframe(tab, use_container_width=True, hide_index=True, height=min(420, 38 * (len(tab) + 1) + 3))
    caption("CX satisfaction is only available for respondents who completed an online booking and rated the experience; blank means fewer than 10 ratings.")

section_title("6. Customer experience ratings (online bookers)")
cx_sel, cx_all = cx_table(full(sub)), cx_table(f)
if cx_sel["n"].max() < 10:
    empty_note("Fewer than 10 online bookers rated the experience in this selection.")
else:
    cats = cx_sel["Touchpoint"].tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=cx_all["Satisfied %"].tolist() + [cx_all["Satisfied %"].iloc[0]], theta=cats + [cats[0]], name="All filtered respondents",
                                  line=dict(color=HALFBAKED, dash="dot"), fill="none"))
    fig.add_trace(go.Scatterpolar(r=cx_sel["Satisfied %"].tolist() + [cx_sel["Satisfied %"].iloc[0]], theta=cats + [cats[0]], name="Current drill-down",
                                  line=dict(color=MIDNIGHT), fill="toself", fillcolor="rgba(0,119,173,0.20)"))
    style_fig(fig, 380)
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], ticksuffix="%")))
    r1, r2 = st.columns([1.4, 1])
    with r1:
        st.plotly_chart(fig, use_container_width=True)
    with r2:
        show = cx_sel.copy()
        show["Satisfied %"] = show["Satisfied %"].round(1)
        show["Mean (1-5)"] = show["Mean (1-5)"].round(2)
        st.dataframe(show, use_container_width=True, hide_index=True)
        caption("Satisfied = rated 4 or 5. Ratings are uniformly high, so treat small differences with caution.")


# ---------------------------------------------------------------- insights
section_title("7. Strategic insights")
na = "Not enough respondents in this selection to say."
lga_g = f.groupby("lga").agg(n=("aware_online", "count"), aware=("aware_online", "mean"))
lga_g = lga_g[lga_g["n"] >= MIN_N].sort_values("aware", ascending=False)
steps = funnel_counts(df)
att_rate = steps[2][1] / steps[1][1] * 100 if steps[1][1] else float("nan")
comp_rate = steps[3][1] / steps[2][1] * 100 if steps[2][1] else float("nan")
ab_s, ab_n = flag_share(f[(f["started_not_completed"] == 1) & (f["completed_booking"] != 1)], "abandon__")
ab_s = ab_s[[i for i in ab_s.index if i not in ("Another reason", "Do not remember")]]
dc_s, dc_n = flag_share(f, "disc__")
dc_s = dc_s[[i for i in dc_s.index if i != "Another channel"]]
cxt = cx_table(f)
cxt = cxt[cxt["n"] >= 10].sort_values("Satisfied %", ascending=False)
ins = [
    insight_card("01", "Awareness varies by area",
                 (f"{lga_g.index[0]} leads with {fmt(lga_g.iloc[0]['aware'] * 100)} aware (n={int(lga_g.iloc[0]['n'])}), while {lga_g.index[-1]} trails at "
                  f"{fmt(lga_g.iloc[-1]['aware'] * 100)} (n={int(lga_g.iloc[-1]['n'])}). Target awareness activity at the low-awareness areas first." if len(lga_g) >= 2 else na), "Geography"),
    insight_card("02", "Where the funnel leaks",
                 (f"Of {steps[1][1]:,} aware respondents, {fmt(att_rate)} attempted a booking and {fmt(comp_rate)} of those who attempted completed one."
                  + (f" The most common reason for not finishing is \"{ab_s.index[0]}\" ({fmt(ab_s.iloc[0])}, small base of {ab_n})." if ab_n >= 10 and len(ab_s) else "")
                  if steps[1][1] else na), "Funnel"),
    insight_card("03", "How people want to hear about us",
                 (f"\"{dc_s.index[0]}\" is the top channel ({fmt(dc_s.iloc[0])}), followed by \"{dc_s.index[1]}\" ({fmt(dc_s.iloc[1])}). "
                  "Lead awareness work through the channels customers already trust." if len(dc_s) >= 2 else na), "Channels"),
    insight_card("04", "Experience once booked",
                 (f"{cxt.iloc[0]['Touchpoint']} rates highest ({fmt(cxt.iloc[0]['Satisfied %'])} satisfied) and {cxt.iloc[-1]['Touchpoint'].lower()} lowest "
                  f"({fmt(cxt.iloc[-1]['Satisfied %'])}), based on {int(cxt['n'].max())} online bookers. Scores are uniformly high, so the gap between steps is small." if len(cxt) >= 2 else na), "Experience"),
]
show_insights(ins, per_row=4)


nav_footer()
