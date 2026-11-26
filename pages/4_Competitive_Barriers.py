# pages/4_Competitive_Barriers.py
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (BLUE_MED, CERULEAN, HALFBAKED, MIDNIGHT, caption, cat_share, csat_proxy, empty_note, filter_bar, flag_share, fmt,
                   full, insight_card, nav_footer, page_header, pct_of, require_data, section_title, setup_page, show_insights, style_fig, top_barrier)

setup_page("Competitive & Barriers")
raw = require_data()
page_header("Competitive & Barriers Intelligence", "Who customers use today, what holds back online booking, and what builds trust")
df = filter_bar(raw)
f = full(df)
if len(f) < 30:
    empty_note("Too few respondents match the current filters.")
    nav_footer()
    st.stop()


def hbar(series, title, color=CERULEAN, height=360, top=None):
    s = series.head(top) if top else series
    s = s.iloc[::-1]
    fig = go.Figure(go.Bar(y=[str(i) for i in s.index], x=s.values, orientation="h", marker_color=color,
                           text=[f"{v:.0f}%" for v in s.values], textposition="outside", cliponaxis=False))
    style_fig(fig, height, legend=False)
    fig.update_layout(title=title, xaxis=dict(visible=False), margin=dict(l=10, r=30, t=40, b=10))
    return fig


# ---------------------------------------------------------------- provider share
section_title("1. Provider usage share (laboratory used for the most recent test)")
named_all = raw[(raw["provider"].notna()) & (raw["provider"] != "(No name given)")]
geo = pd.crosstab(named_all["provider"], named_all["region"])
named = df[(df["provider"].notna()) & (df["provider"] != "(No name given)")]
counts = named["provider"].value_counts()
syn_n = int(counts.get("SYNLAB", 0))
if counts.empty:
    empty_note("No laboratory names in this selection.")
else:
    top = counts.head(12)
    plot = top.copy()
    if "SYNLAB" not in plot.index and syn_n > 0:
        plot.loc["SYNLAB"] = syn_n


    def geo_class(lab):
        a = geo.loc[lab].get("FCT Abuja", 0) > 0 if lab in geo.index else False
        l = geo.loc[lab].get("Lagos", 0) > 0 if lab in geo.index else False
        return "Both states" if a and l else ("Abuja-only" if a else "Lagos-only")


    cmap = {"Abuja-only": CERULEAN, "Lagos-only": HALFBAKED, "Both states": BLUE_MED}
    labs = list(plot.index)[::-1]
    colors = [MIDNIGHT if l == "SYNLAB" else cmap[geo_class(l)] for l in labs]
    share = plot / len(named) * 100
    fig = go.Figure(go.Bar(y=[("► " + l if l == "SYNLAB" else l) for l in labs], x=[plot[l] for l in labs], orientation="h", marker_color=colors,
                           text=[f"{plot[l]} ({share[l]:.1f}%)" for l in labs], textposition="outside", cliponaxis=False))
    style_fig(fig, max(340, 28 * len(labs) + 70), legend=False)
    fig.update_layout(title="Mentions by laboratory", xaxis=dict(visible=False), margin=dict(l=10, r=60, t=40, b=10))
    k1, k2 = st.columns([2.2, 1])
    with k1:
        st.plotly_chart(fig, use_container_width=True)
    with k2:
        rank = int((counts > syn_n).sum()) + 1 if syn_n else None
        tied = int((counts == syn_n).sum()) - 1 if syn_n else 0
        st.markdown(f"**{len(named):,}** respondents named a lab; **{counts.size}** different labs.")
        st.markdown(f"Leader: **{counts.index[0]}** with {counts.iloc[0]} mentions ({counts.iloc[0] / len(named) * 100:.0f}% of named).")
        if syn_n:
            st.markdown(f"**SYNLAB:** {syn_n} mentions ({syn_n / len(named) * 100:.1f}%), rank {rank}" + (f", tied with {tied} other lab(s)." if tied else "."))
        else:
            st.markdown("**SYNLAB:** not named in this selection.")
        st.markdown("<span style='font-size:12px;color:#64748B;'>Bar colours: dark blue = SYNLAB; mid blue = lab named in both states; "
                    "cerulean = Abuja-only; light blue = Lagos-only (by where respondents live).</span>", unsafe_allow_html=True)
    caption("Lab names are free text, grouped by hand into brands. Location is where the respondent lives, not where the lab is.")

# ---------------------------------------------------------------- barriers
c1, c2 = st.columns(2)
with c1:
    section_title("2. Why people have not booked online")
    s, n = flag_share(f, "never__")
    if n < 10:
        empty_note()
    else:
        s = s[[i for i in s.index if i != "Another reason"]]
        st.plotly_chart(hbar(s, "Reasons for not booking online", MIDNIGHT, 400), use_container_width=True)
        caption(f"Base: {n} respondents who did not complete an online booking. Multiple answers allowed.")
with c2:
    section_title("3. Why started bookings were abandoned")
    s, n = flag_share(f[(f["started_not_completed"] == 1) & (f["completed_booking"] != 1)], "abandon__")
    if n < 10:
        empty_note(f"Only {n} respondents abandoned a booking in this selection.")
    else:
        s = s[[i for i in s.index if i not in ("Another reason", "Do not remember")]]
        st.plotly_chart(hbar(s, "Reasons for abandoning a booking", BLUE_MED, 400), use_container_width=True)
        caption(f"Base: {n} respondents. Small base: directional only.")

# ---------------------------------------------------------------- trust drivers + results
c3, c4 = st.columns(2)
with c3:
    section_title("4. What matters when choosing an online lab")
    s3, n3 = flag_share(f, "prio__")
    s3 = s3[[i for i in s3.index if i not in ("Another priority", "Not sure")]]
    s1 = cat_share(f["top_priority"]).reindex(s3.index).fillna(0)
    order = list(s3.index)[::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=order, x=[s3[o] for o in order], orientation="h", name="In top three", marker_color=HALFBAKED,
                         text=[f"{s3[o]:.0f}%" for o in order], textposition="outside", cliponaxis=False))
    fig.add_trace(go.Bar(y=order, x=[s1[o] for o in order], orientation="h", name="Single most important", marker_color=MIDNIGHT,
                         text=[f"{s1[o]:.0f}%" for o in order], textposition="outside", cliponaxis=False))
    style_fig(fig, 460)
    fig.update_layout(title="Trust and choice drivers", barmode="group", xaxis=dict(visible=False), margin=dict(l=10, r=40, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    caption(f"Base: {n3} respondents. Pick up to three, then the single most important.")
with c4:
    section_title("5. Preferred way to receive results")
    s = cat_share(f["results_pref"]).sort_values(ascending=False)
    short = {"A printed copy collected from the laboratory": "Printed copy from the lab", "A secure patient portal or app": "Secure portal or app",
             "A secure link sent by WhatsApp": "Secure link by WhatsApp", "A secure link sent by email": "Secure link by email",
             "A secure link sent by SMS": "Secure link by SMS", "An authorised person helping me access them": "Authorised person helps"}
    s.index = [short.get(i, i) for i in s.index]
    st.plotly_chart(hbar(s, "Results delivery preference", CERULEAN, 460), use_container_width=True)

# ---------------------------------------------------------------- SWOT
section_title("6. SWOT grid for the digital offering (analyst interpretation of this selection)")
csat, _n_cx = csat_proxy(f)
digital = (df["digital_access"] == "I can do this myself").mean() * 100
aware = pct_of(f["aware_online"])
disc, _ = flag_share(f, "disc__")
doc = disc.get("A doctor or pharmacist", float("nan"))
ch = cat_share(f["booking_channel"])
wa = ch.get("WhatsApp or another social media service", float("nan"))
tb_label, tb_rate, _ = top_barrier(f)
leader_share = counts.iloc[0] / len(named) * 100 if len(named) else float("nan")
s_col, w_col = st.columns(2)
swot = {
    "s": ("Strengths", [f"Online bookers are highly satisfied ({fmt(csat)} rate steps 4 or 5; small base, uniform scores)",
                         f"{fmt(digital)} can already book or pay online themselves",
                         "SYNLAB is a recognised brand when prompted"]),
    "w": ("Weaknesses", [f"Only {fmt(aware)} know a test can be booked online",
                          f"SYNLAB is rarely named as the lab used ({syn_n} mentions)",
                          f"Top stated barrier: \"{tb_label}\" ({fmt(tb_rate)})"]),
    "o": ("Opportunities", [f"{fmt(doc)} want to hear about services from a doctor or pharmacist: referral partnerships",
                             f"WhatsApp and social media already account for {fmt(wa)} of recent online bookings",
                             "Home collection and HMO / employer billing for niche segments"]),
    "t": ("Threats", [f"The leading lab holds {fmt(leader_share)} of named mentions in this selection; the rest is fragmented",
                       "Strong preference for in-person arrangements and printed results",
                       "Stated intent overstates real behaviour; price and trust claims need proof"]),
}
for col, keys in [(s_col, ["s", "o"]), (w_col, ["w", "t"])]:
    with col:
        for k in keys:
            title, items = swot[k]
            st.markdown(f'<div class="swot-card swot-{k}" style="margin-bottom:12px;"><h4>{title}</h4><ul>' + "".join(f"<li>{i}</li>" for i in items) + "</ul></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- insights
section_title("7. Strategic insights")
na = "Not enough respondents in this selection to say."
rank_txt = ""
if syn_n and len(counts):
    rank_txt = f" SYNLAB has {syn_n} mention{'s' if syn_n != 1 else ''} (joint rank {int((counts > syn_n).sum()) + 1})."
nb_s, nb_n = flag_share(f, "never__")
not_needed = next((v for k, v in nb_s.items() if k.startswith("I have not needed")), float("nan"))
unaware = next((v for k, v in nb_s.items() if k.startswith("I did not know")), float("nan"))
res = cat_share(f["results_pref"])
res_digital = float(res[[i for i in res.index if i.startswith("A secure")]].sum()) if len(res) else float("nan")
res_print = float(res[[i for i in res.index if i.startswith("A printed")]].sum()) if len(res) else float("nan")
info = cat_share(f["info_pref"])
ins = [
    insight_card("01", "A fragmented market",
                 (f"{counts.size} different labs were named by {len(named):,} respondents. The leader, {counts.index[0]}, holds {fmt(counts.iloc[0] / len(named) * 100)} of mentions."
                  + rank_txt if len(counts) else na), "Competition"),
    insight_card("02", "What holds back online booking",
                 (f"The most-cited barrier is \"{tb_label}\" ({fmt(tb_rate)}). A further {fmt(not_needed)} simply had not needed a test"
                  + ("." if tb_label.startswith("I did not know") else f" and {fmt(unaware)} did not know booking online was possible.")
                  if nb_n >= 10 else na), "Barriers"),
    insight_card("03", "Trust drives choice",
                 (f"{s3.index[0]} is in the top three for {fmt(s3.iloc[0])} and is the single most important factor for {fmt(s1.get(s3.index[0], float('nan')))}. "
                  + (f"The most helpful information is {info.index[0].lower()} ({fmt(info.iloc[0])})." if len(info) else "") if len(s3) else na), "Trust"),
    insight_card("04", "Results delivery",
                 (f"{fmt(res_digital)} prefer a secure digital route (portal, WhatsApp, email or SMS) while {fmt(res_print)} still want a printed copy from the lab. "
                  "Offer both." if len(res) else na), "Results"),
]
show_insights(ins, per_row=4)


nav_footer()
