"""
prepare_data.py - turn the Kobo export into the analysis file the dashboard reads.

Usage
-----
python prepare_data.py \
    --clean SYNLAB_survey_deduplicated_completes.csv \
    --all   SYNLAB_survey_deduplicated.csv \
    --raw   "<original Kobo export>.csv" \
    --out   data/synlab_dashboard.csv

--clean : the de-duplicated ELIGIBLE respondents (semicolon-delimited, label headers)
--all   : the de-duplicated file with all respondents (used for the total shown on the cover)
--raw   : the original export (only used for the count of submissions received)
"""
import argparse, difflib, json, os, re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- helpers
def first(df, prefix, exact=False):
    """First column whose header starts with `prefix` (sub-option columns 'q/option' are skipped
    unless the prefix itself contains a slash)."""
    for c in df.columns:
        if "/" in c and "/" not in prefix:
            continue
        if c.startswith(prefix):
            return c
    raise KeyError(prefix)

def multi(df, stem, rename=None):
    """Return {short_label: 0/1 float Series} for a Kobo multi-select 'stem/option' block."""
    out = {}
    for c in df.columns:
        if c.startswith(stem + "/"):
            lab = c.split("/", 1)[1]
            out[lab] = pd.to_numeric(df[c], errors="coerce")
    return out

RATING = {"Very dissatisfied": 1, "Dissatisfied": 2, "Neither satisfied nor dissatisfied": 3,
          "Satisfied": 4, "Very satisfied": 5}

PROVIDER_RULES = [
    ("Garki Medical Laboratory/Hospital", r"garki"), ("Echolab", r"echolab"), ("Echoscan Service Ltd", r"echoscan"),
    ("Labcrest", r"labcrest"), ("Firmcare", r"firmcare"), ("Beaconhealth", r"beacon\s*health"), ("Aafiya", r"aafiya"),
    ("Lifepro Medicare", r"lifepro"), ("Lion Medical", r"^lion "), ("SYNLAB", r"syn\s*lab|sin\s*lab|synlap"),
    ("Everight", r"everight"), ("Lab360", r"lab\s*360"), ("Zeda", r"^zeda"), ("Vinette", r"vinette"),
    ("Body Affairs", r"body affairs"), ("Atlas Medical", r"^atlas"), ("E-Clinic", r"e-clinic"),
    ("Marie Stopes", r"marie stopes"), ("Premier Care", r"premier care"), ("Peak Diagnostic", r"^peak "),
    ("Celib", r"^celib"), ("Medprime", r"medprime"), ("Afriglobal", r"afriglobal"), ("Adefemi Hospital", r"adefemi"),
    ("Altaa Health", r"altaa"), ("Value Care", r"value care"), ("Healthpro", r"healthpro"), ("Shalom", r"shalom"),
    ("Image Planet", r"image planet"), ("Medplus", r"medplus"), ("Union Diagnostic", r"^union diag"),
    ("Biosensors", r"biosensors"), ("Agecare", r"agecare"), ("Oris", r"^oris"), ("JCP Max", r"jcp max"),
    ("Patho-medics", r"patho"), ("LASUTH", r"lasuth"), ("LUTH", r"^luth$"), ("Experior", r"experior"),
    ("Clina Lancet", r"clina"), ("Cellulose", r"cellulose"), ("Lifebridge", r"lifebridge"), ("Sura Health", r"^sura"),
    ("A.R. Ultimate", r"a\.r\. ultimate"), ("Medcrux", r"medcrux"), ("Tender Love Medcare", r"tender love"),
    ("Eagleview", r"eagleview"), ("Clinix", r"clinix"), ("Biopath", r"biopath"), ("National Hospital Abuja", r"^national hospital"),
]
NO_NAME = r"remember|prefer not|can[’']t say|didn[’']t know|^school$|^a private hospital lab$|^teaching hospital lab|^(the )?hospital lab$|^i do not"

def canon_provider(x):
    if pd.isna(x): return np.nan
    l = str(x).strip().lower()
    for name, pat in PROVIDER_RULES:
        if re.search(pat, l): return name
    if re.search(NO_NAME, l): return "(No name given)"
    return str(x).strip().title()

# ---------------------------------------------------------------- main build
def build(clean_path):
    df = pd.read_csv(clean_path, sep=";", dtype=str)
    o = pd.DataFrame(index=df.index)
    o["id"] = df["_id"]
    o["full_survey"] = (df["digital_ok"] == "1").astype(int)       # passed the digital-access gate
    o["region"] = df[first(df, "Where do you currently live?")]
    o["lga"] = df[first(df, "Which LGA or Area Council")]
    o["neighbourhood"] = df[first(df, "Which neighbourhood")].str.strip().str.title()
    o["age"] = df[first(df, "Which age group")]
    role = df[first(df, "When laboratory tests are needed, do you help choose")]
    o["role"] = role
    o["healthcare_role"] = pd.to_numeric(df[first(df, "Which of the following describes your work or involvement?/I work in another healthcare role")], errors="coerce")
    o["gender"] = df[first(df, "What is your gender?")]
    o["employment"] = df[first(df, "What is your main employment status?")]
    ins = df[first(df, "Do you currently have health insurance or HMO coverage?", exact=True)]
    o["insurance_raw"] = ins
    o["insurance"] = np.select([ins.isna(), ins.fillna("").str.startswith("Yes"), ins == "No"],
                               [None, "Insured", "Uninsured"], default="Not sure / no answer")
    o["insurance"] = o["insurance"].where(ins.notna())
    # testing behaviour
    o["tests_12m"] = df[first(df, "In the past 12 months, how many times have you had a laboratory test")]
    o["had_test"] = pd.to_numeric(df["had_test"], errors="coerce")
    o["test_for"] = df[first(df, "Who was the most recent test for?")]
    o["test_reason"] = df[first(df, "What was the main reason for that test?")]
    o["lab_chooser"] = df[first(df, "Who mainly decided which laboratory")]
    o["provider_raw"] = df[first(df, "Which laboratory or provider did you use")]
    o["provider"] = o["provider_raw"].map(canon_provider)
    for lab, s in multi(df, first(df, "Who paid for the test?", exact=True)).items():
        o["paid_" + re.sub(r"\W+", "_", lab.lower())[:28].strip("_")] = s
    o["spend_band"] = df[first(df, "Approximately how much did you personally pay")]
    # digital
    o["digital_access"] = df[first(df, "How would you usually access an online service")]
    o["aware_online"] = (df[first(df, "Before this survey, did you know it was possible")] == "Yes").astype(float).where(o["full_survey"] == 1)
    adopt = multi(df, first(df, "In the past 12 months, which of these have you personally done", exact=True))
    amap = {"Completed an online booking and paid online": "booked_paid_online",
            "Completed an online booking but paid at collection or at the laboratory": "booked_paid_lab",
            "Completed an online booking paid for by an HMO, employer or another person": "booked_third_party_paid",
            "Started an online booking or payment but did not complete it": "started_not_completed",
            "None of these": "no_online_action", "Do not remember": "dont_remember_online"}
    for lab, s in adopt.items():
        o[amap[lab]] = s
    o["completed_booking"] = o[["booked_paid_online", "booked_paid_lab", "booked_third_party_paid"]].max(axis=1)
    o["attempted_booking"] = o[["booked_paid_online", "booked_paid_lab", "booked_third_party_paid", "started_not_completed"]].max(axis=1)
    o["booking_channel"] = df[first(df, "Where did you complete your most recent online booking?")]
    o["collection_location"] = df[first(df, "For that booking, where was the sample collected?")]
    for col, nm in zip(["Finding and understanding the test or package", "Completing the booking", "Making payment, if you paid",
                        "Sample collection, if completed", "Time taken to receive results, if received"],
                       ["cx_finding", "cx_booking", "cx_payment", "cx_collection", "cx_results"]):
        o[nm] = df[first(df, col)].map(RATING)
    o["improvement_text"] = df[first(df, "What is the one improvement that would most improve")]
    for lab, s in multi(df, first(df, "What were the main reasons you did not complete an online booking or payment?", exact=True)).items():
        o["abandon__" + lab] = s
    for lab, s in multi(df, first(df, "What are the main reasons you have not completed an online laboratory booking", exact=True)).items():
        o["never__" + lab] = s
    # preferences
    for lab, s in multi(df, first(df, "Which three things matter most when choosing an online laboratory service?", exact=True)).items():
        o["prio__" + lab] = s
    o["top_priority"] = df[first(df, "Which ONE of your selections matters most?")]
    o["collect_pref"] = df[first(df, "For a test that can be collected either way")]
    o["pay_pref"] = df[first(df, "Which payment arrangement would you prefer?")]
    for lab, s in multi(df, first(df, "For yourself or someone you help, which services might be relevant", exact=True)).items():
        o["svc__" + lab] = s
    o["results_pref"] = df[first(df, "How would you prefer to access your results?")]
    for lab, s in multi(df, first(df, "Where would you prefer to hear about laboratory services?", exact=True)).items():
        o["disc__" + lab] = s
    o["info_pref"] = df[first(df, "Which one type of information would most help you decide")]
    # price test
    o["offer_tier"] = df["offer_tier"]
    o["offer_price"] = pd.to_numeric(df["offer_price"], errors="coerce")
    intent = df[first(df, "How likely would you be to book this package online")]
    o["intent"] = intent
    o["intent_top2"] = intent.isin(["Definitely would", "Probably would"]).astype(float).where(intent.notna())
    o["intent_definite"] = (intent == "Definitely would").astype(float).where(intent.notna())
    need = df[first(df, "How likely is it that you or someone you arrange tests for will actually need")]
    o["need_likelihood"] = need
    o["need_flag"] = need.isin(["Very likely", "Somewhat likely"]).astype(float).where(need.notna())
    o["price_reason_text"] = df[first(df, "What is the main reason for your answer about booking at this price?")]
    o["home_fee_choice"] = df[first(df, "If the same tests could be collected at home")]
    o["home_fee_ngn"] = pd.to_numeric(df[first(df, "Please enter the amount in Naira")], errors="coerce")
    o["synlab_used"] = df[first(df, "Before today, had you or someone you arrange tests for used SYNLAB?")]
    o["comment_text"] = df[first(df, "Is there anything else you would like us to understand")]
    o["followup_consent"] = df[first(df, "May the research team contact you for one follow-up")]
    return o

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True, help="de-duplicated ELIGIBLE respondents (e.g. ..._completes.csv)")
    ap.add_argument("--all", help="de-duplicated file with ALL respondents (eligible and screened out); used for the total")
    ap.add_argument("--raw", help="original Kobo export (only used for the raw submission count)")
    ap.add_argument("--out", default="data/synlab_dashboard.csv")
    a = ap.parse_args()
    out = build(a.clean)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"Wrote {a.out}: {len(out)} rows x {out.shape[1]} columns")

    # respondent-base counts shown on the cover page
    summary = {"eligible": int(len(out)), "full_survey": int(out["full_survey"].sum())}
    if a.all:
        summary["total_respondents"] = int(len(pd.read_csv(a.all, sep=";", dtype=str, usecols=["_id"])))
    if a.raw:
        summary["raw_submissions"] = int(len(pd.read_csv(a.raw, sep=";", dtype=str, usecols=["_id"])))
        if "total_respondents" in summary:
            summary["duplicates_removed"] = summary["raw_submissions"] - summary["total_respondents"]
    spath = os.path.join(os.path.dirname(a.out) or ".", "survey_summary.json")
    with open(spath, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("Wrote", spath, summary)
