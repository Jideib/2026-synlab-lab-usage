"""Cover / landing page. Streamlit treats this as the root route."""
import streamlit as st

from common import LOGO, banner, bootstrap, card, kpi, sec


d, c, F, FO, FA, base_note = bootstrap()

c1, c2 = st.columns([1, 2])
if LOGO.exists():
    c1.image(str(LOGO), width="stretch")

banner("Laboratory Services Survey",
       "Executive dashboard · Lagos and FCT Abuja · November 2026")

k = st.columns(4)
kpi(k[0], "1,373", "Submissions", "in the export")
kpi(k[1], f"{len(d):,}", "Eligible completes", "after screening and dedupe")
kpi(k[2], f"{int(d['online'].sum()):,}", "Online-ready",
    "answered the online block")
kpi(k[3], f"{(d.city == 'Lagos').sum()} / {(d.city == 'Abuja').sum()}",
    "Lagos / Abuja", "among eligible completes")

sec("What is inside")
for title, body in [
    ("Overview", "Headline metrics, screening funnel and Lagos vs Abuja."),
    ("Brand Health", "SYNLAB use, drivers of lab choice, providers named."),
    ("Customer Insights", "Who arranges tests, why, who pays, preferences."),
    ("Competitive Intelligence",
     "Online booking behaviour, barriers, satisfaction, channels."),
    ("Strategic Analytics",
     "Price response, intent model, customer segments, priorities."),
]:
    card(title, body)

sec("Data quality checks")
q = st.columns(3)
with q[0]:
    dups = int(d["_uuid"].duplicated().sum()) if "_uuid" in d else 0
    kpi(q[0], f"{dups}", "Duplicate UUIDs", "should be 0")
with q[1]:
    miss = int(d["price_label"].isna().sum())
    kpi(q[1], f"{miss}", "Missing offer price", "excluded from intent model")
with q[2]:
    unk = int((d["insured"] == "Unknown").sum())
    kpi(q[2], f"{unk}", "Insurance unknown", "shown separately in model")
