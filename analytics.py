# analytics.py - logistic regression and k-means, implemented with numpy/pandas only
# (no extra dependencies; results are checked against scikit-learn in testing).
import math

import numpy as np
import pandas as pd

AGE_GROUPS = {"18-24": "Age 18-24", "35-44": "Age 35-44", "45-54": "Age 45-54", "55-64": "Age 55+", "65 or above": "Age 55+"}
EMP_GROUPS = {"Self-employed or business owner": "Self-employed",
              "Not currently in paid employment": "Not employed, student or retired", "Student": "Not employed, student or retired",
              "Retired": "Not employed, student or retired", "Another status": "Other / not stated", "Prefer not to say": "Other / not stated"}

OUTCOMES = {
    "Completed an online booking (past 12 months)": "completed_booking",
    "Would book at the price shown (definitely or probably)": "intent_top2",
}


# ------------------------------------------------------------------ logistic regression
def design_matrix(df, outcome, include_price=False):
    """Build the design matrix. Reference groups: Abuja, age 25-34, uninsured, employed, can book online alone, man."""
    d = df.copy()
    X = pd.DataFrame(index=d.index)
    X["Lagos (vs Abuja)"] = (d["region"] == "Lagos").astype(float)
    for raw_lab, lab in {k: v for k, v in AGE_GROUPS.items()}.items():
        X[lab] = X.get(lab, 0.0) + (d["age"] == raw_lab).astype(float)
    X["Insured (vs uninsured)"] = (d["insurance"] == "Insured").astype(float)
    X["Insurance not stated"] = (~d["insurance"].isin(["Insured", "Uninsured"])).astype(float)
    for raw_lab, lab in EMP_GROUPS.items():
        X[lab] = X.get(lab, 0.0) + (d["employment"] == raw_lab).astype(float)
    X["Employment not stated"] = d["employment"].isna().astype(float)
    X["Needs help to book online"] = (d["digital_access"] == "I usually need someone to help me").astype(float)
    X["Someone else books for them"] = (d["digital_access"] == "Someone else does it for me").astype(float)
    X["Woman (vs man)"] = (d["gender"] == "Woman").astype(float)
    X["Gender not stated / other"] = (~d["gender"].isin(["Woman", "Man"])).astype(float)
    if include_price:
        X["Price shown (per ₦10,000)"] = d["offer_price"] / 10000.0
    y = pd.to_numeric(d[outcome], errors="coerce")
    ok = y.notna() & X.notna().all(axis=1)
    X, y = X[ok], y[ok]
    # drop constant columns (e.g. when one region is selected) and groups too small to estimate
    keep = [c for c in X.columns if X[c].nunique() > 1 and X[c].sum() >= 5 and (len(X) - X[c].sum()) >= 5 or
            (c.startswith("Price") and X[c].nunique() > 1)]
    return X[keep], y


def _p_from_z(z):
    return math.erfc(abs(z) / math.sqrt(2))


def auc_score(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank().to_numpy()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def logit_fit(X, y, max_iter=100, tol=1e-9, ridge=1e-8):
    """Unpenalised logistic regression by IRLS. Returns (coefficient table, fit statistics)."""
    names = ["Intercept"] + list(X.columns)
    A = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    yy = np.asarray(y, dtype=float)
    if yy.sum() < 10 or (len(yy) - yy.sum()) < 10:
        raise ValueError("Too few cases in one outcome group to estimate a model.")
    beta = np.zeros(A.shape[1])
    for _ in range(max_iter):
        eta = np.clip(A @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        w = p * (1 - p)
        H = A.T @ (A * w[:, None]) + ridge * np.eye(A.shape[1])
        step = np.linalg.solve(H, A.T @ (yy - p))
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    eta = np.clip(A @ beta, -30, 30)
    p = 1 / (1 + np.exp(-eta))
    H = A.T @ (A * (p * (1 - p))[:, None]) + ridge * np.eye(A.shape[1])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    z = beta / se
    tab = pd.DataFrame({"Term": names, "coef": beta, "se": se, "z": z, "p": [_p_from_z(v) for v in z]})
    tab["OR"] = np.exp(tab["coef"])
    tab["OR_low"] = np.exp(tab["coef"] - 1.96 * tab["se"])
    tab["OR_high"] = np.exp(tab["coef"] + 1.96 * tab["se"])
    tab["unstable"] = (tab["se"] > 5) | (tab["coef"].abs() > 12)
    ll = float(np.sum(yy * np.log(np.clip(p, 1e-12, 1)) + (1 - yy) * np.log(np.clip(1 - p, 1e-12, 1))))
    p0 = yy.mean()
    ll0 = float(len(yy) * (p0 * math.log(p0) + (1 - p0) * math.log(1 - p0)))
    stats = {"n": int(len(yy)), "events": int(yy.sum()), "rate": float(p0), "r2": 1 - ll / ll0, "auc": auc_score(yy, p),
             "accuracy": float(((p >= 0.5) == (yy == 1)).mean()), "baseline_accuracy": float(max(p0, 1 - p0))}
    return tab, stats


def auc_label(a):
    if a != a: return "n/a"
    return "poor" if a < 0.6 else "modest" if a < 0.7 else "acceptable" if a < 0.8 else "strong"


def _effect_text(r):
    ci = f"OR {r['OR']:.2f}, 95% CI {r['OR_low']:.2f} to {r['OR_high']:.2f}"
    if r["Term"].startswith("Price"):
        return f"<b>{r['Term']}</b>: each extra ₦10,000 changes the odds by {abs(r['OR'] - 1) * 100:.0f}% {'down' if r['OR'] < 1 else 'up'} ({ci})."
    if r["OR"] >= 1:
        return f"<b>{r['Term']}</b>: odds about {r['OR']:.1f} times those of the reference group ({ci}), other factors held equal."
    return f"<b>{r['Term']}</b>: odds about {(1 - r['OR']) * 100:.0f}% lower than the reference group ({ci}), other factors held equal."


def interpret_logit(tab, stats, outcome_label):
    """Plain-language reading of the model (statistical wording). Not used by the page once executive_summary is wired in."""
    lines = [f"The model uses <b>{stats['n']:,}</b> respondents; <b>{stats['events']:,}</b> ({stats['rate'] * 100:.0f}%) "
             f"are in the outcome group: <i>{outcome_label.lower()}</i>."]
    a = stats["auc"]
    fit = f"Fit is <b>{auc_label(a)}</b> (AUC {a:.2f}, McFadden R² {stats['r2']:.2f})."
    if a < 0.8:
        fit += " The factors explain only part of who does this, so read them as tendencies, not rules."
    lines.append(fit)
    terms = tab[tab["Term"] != "Intercept"].copy()
    placeholder = terms["Term"].str.contains("not stated")
    main = terms[~placeholder]
    sig = main[(main["p"] < 0.05) & (~main["unstable"])].sort_values("OR", ascending=False)
    for _, r in sig.iterrows():
        lines.append(_effect_text(r))
    if sig.empty:
        lines.append("No factor shows a clear independent effect at the 5% level.")
    ns = main[main["p"] >= 0.05]["Term"].tolist()
    if ns:
        lines.append("No clear independent effect (p of 0.05 or more): " + ", ".join(ns) + ".")
    return lines


# ------------------------------------------------------------------ executive summary (business language)
_WHO = {  # factor -> (who, compared with)
    "Lagos (vs Abuja)": ("Lagos respondents", "Abuja respondents"),
    "Age 18-24": ("18 to 24 year-olds", "25 to 34 year-olds"),
    "Age 35-44": ("35 to 44 year-olds", "25 to 34 year-olds"),
    "Age 45-54": ("45 to 54 year-olds", "25 to 34 year-olds"),
    "Age 55+": ("Respondents aged 55 and over", "25 to 34 year-olds"),
    "Insured (vs uninsured)": ("Insured customers", "uninsured customers"),
    "Self-employed": ("Self-employed respondents", "employed respondents"),
    "Not employed, student or retired": ("People not in paid work, students and retirees", "employed respondents"),
    "Needs help to book online": ("People who need help to book online", "people who book online alone"),
    "Someone else books for them": ("People who rely on someone else to book", "people who book online alone"),
    "Woman (vs man)": ("Women", "men"),
}


def average_effects(X, tab):
    """Average change, in percentage points, in the chance of the outcome when a factor switches from 0 to 1
    (price: per extra NGN 10,000), with all other factors left as observed."""
    beta = tab.set_index("Term")["coef"]
    cols = ["Intercept"] + list(X.columns)
    b = beta.reindex(cols).to_numpy()
    base = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    out = {}
    for j, c in enumerate(X.columns, start=1):
        hi, lo = base.copy(), base.copy()
        if c.startswith("Price"):
            hi[:, j] = base[:, j] + 1
        else:
            hi[:, j], lo[:, j] = 1, 0
        out[c] = float((sig(hi @ b) - sig(lo @ b if not c.startswith("Price") else base @ b)).mean() * 100)
    return out


def executive_summary(X, tab, stats, outcome_key):
    """Return [(heading, [sentences])] describing the model for executives, without statistical jargon."""
    booking = outcome_key == "completed_booking"
    eff = average_effects(X, tab)
    p = tab.set_index("Term")["p"]
    unstable = tab.set_index("Term")["unstable"]
    verb = "to have booked online" if booking else "to say they would book at the price shown"
    clear = [t for t in eff if t in _WHO and p[t] < 0.05 and not unstable[t]]
    clear.sort(key=lambda t: -abs(eff[t]))
    unclear = [t for t in eff if t in _WHO and t not in clear]

    intro = ("We compared who has already booked a lab test online, looking at region, age, insurance, employment and digital habits together, "
             "so each difference below holds the other factors equal." if booking else
             "We compared who says they would book the Basic Wellness Panel at the price they were shown, looking at region, age, insurance, "
             "employment and digital habits together, so each difference below holds the other factors equal.")
    sections = [("What we looked at", [intro])]

    stand = []
    for t in clear:
        who, vs = _WHO[t]
        stand.append(f"<b>{who}</b> are about {abs(eff[t]):.0f} points {'more' if eff[t] > 0 else 'less'} likely {verb} than {vs}.")
    price_t = next((t for t in eff if t.startswith("Price")), None)
    if price_t:
        if p[price_t] < 0.05:
            stand.append(f"<b>Price:</b> each extra ₦10,000 {'lowers' if eff[price_t] < 0 else 'raises'} the share who would book by about {abs(eff[price_t]):.0f} points.")
        else:
            stand.append("<b>Price:</b> across the tested range the price shown has no clear effect on intent.")
    if not stand:
        stand = ["No single factor stands out once the others are accounted for."]
    sections.append(("What stands out", stand))
    if unclear:
        sections.append(("No clear difference", [", ".join(_WHO[t][0].lower() if not t.startswith("Lagos") else "region" for t in unclear[:8]) + "."]))

    acc, base = stats["accuracy"] * 100, stats["baseline_accuracy"] * 100
    strength = "useful for direction, not for predicting any one person" if acc - base >= 5 else "only slightly better than guessing, so treat it with caution"
    sections.append(("How reliable is it?", [f"The model sorts respondents correctly {acc:.0f}% of the time, against {base:.0f}% if we simply guessed the most common answer. "
                                             f"It is {strength}."]))

    sowhat = []
    if booking:
        if "Insured (vs uninsured)" in clear and eff["Insured (vs uninsured)"] > 0:
            sowhat.append("Online booking sits mainly with insured customers, so insurers, HMOs and employers are the most direct route to grow it.")
        if "Age 18-24" in clear and eff["Age 18-24"] > 0:
            sowhat.append("Under-25s are the natural early adopters.")
        if any(t in clear and eff[t] > 0 for t in ("Needs help to book online", "Someone else books for them")):
            sowhat.append("People who need help, or rely on someone else, book more often online, probably because an employer, HMO or relative booked for them.")
    else:
        if "Needs help to book online" in clear and eff["Needs help to book online"] < 0:
            sowhat.append("Customers who need help to book online are the biggest gap, so assisted booking by phone or WhatsApp is the clearest opportunity.")
        if price_t and p[price_t] >= 0.05:
            sowhat.append("The tested price is not what holds intent back; ease of booking and customer type matter more.")
        sowhat.append("Stated intent overstates real bookings, so confirm with a pilot before committing budget.")
    if not sowhat:
        sowhat = ["Use these as directional signals and confirm with a pilot."]
    sections.append(("So what", sowhat))

    reg = "Lagos (vs Abuja)"
    if reg in eff and reg in clear and abs(eff[reg]) >= 25:
        sections.append(("Caution", ["The Lagos versus Abuja gap is unusually large. Confirm it with field checks before building strategy on it."]))
    return sections


# ------------------------------------------------------------------ k-means
def _sqdist(X, C):
    d = (X ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2 * X @ C.T
    return np.maximum(d, 0)


def _kmeans_pp(X, k, rng):
    n = len(X)
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(_sqdist(X, np.array(centers)), axis=1)
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        centers.append(X[rng.choice(n, p=probs)])
    return np.array(centers)


def kmeans(X, k, n_init=10, max_iter=200, seed=42):
    """Lloyd's k-means with k-means++ starts. Returns (labels, centers, inertia)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        c = _kmeans_pp(X, k, rng)
        for _ in range(max_iter):
            d = _sqdist(X, c)
            lab = d.argmin(axis=1)
            new = np.array([X[lab == j].mean(axis=0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        d = _sqdist(X, c)
        lab = d.argmin(axis=1)
        inertia = float(d[np.arange(len(X)), lab].sum())
        if best is None or inertia < best[2]:
            best = (lab, c, inertia)
    return best


def silhouette(X, labels):
    n = len(X)
    D = np.sqrt(_sqdist(X, X))
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return float("nan")
    a = np.zeros(n); b = np.full(n, np.inf)
    for u in uniq:
        m = labels == u
        cnt = m.sum()
        a[m] = D[np.ix_(m, m)].sum(axis=1) / max(cnt - 1, 1)
        for v in uniq:
            if v == u: continue
            mv = labels == v
            b[m] = np.minimum(b[m], D[np.ix_(m, mv)].mean(axis=1))
    s = (b - a) / np.maximum(a, b)
    s[np.isnan(s)] = 0
    return float(s.mean())


FEATURES = {
    "age_num": "Older age",
    "digital_self": "Books online alone",
    "insured": "Insured",
    "aware_online": "Aware of online booking",
    "completed_booking": "Has booked online",
    "intent_top2": "Would book at tested price",
    "need_flag": "Likely to need the package",
    "home_pref": "Prefers home collection",
    "tests_num": "Frequent tester",
    "pays_self": "Pays personally",
}
AGE_NUM = {"18-24": 1, "25-34": 2, "35-44": 3, "45-54": 4, "55-64": 5, "65 or above": 6}
TESTS_NUM = {"Once": 1, "2-3 times": 2, "4-5 times": 3, "6 or more times": 4}


def cluster_features(df):
    d = df.copy()
    F = pd.DataFrame(index=d.index)
    F["age_num"] = d["age"].map(AGE_NUM)
    F["digital_self"] = (d["digital_access"] == "I can do this myself").astype(float)
    F["insured"] = (d["insurance"] == "Insured").astype(float)
    F["aware_online"] = d["aware_online"]
    F["completed_booking"] = d["completed_booking"]
    F["intent_top2"] = d["intent_top2"]
    F["need_flag"] = d["need_flag"]
    F["home_pref"] = (d["collect_pref"] == "At home").astype(float)
    F["tests_num"] = d["tests_12m"].map(TESTS_NUM).fillna(0)       # 0 = no test / cannot recall
    F["pays_self"] = d["paid_i_paid"].fillna(0)
    F = F.apply(pd.to_numeric, errors="coerce")
    ok = F.notna().all(axis=1)
    return F[ok]


NEG_LABEL = {"age_num": "Younger", "digital_self": "Needs help online", "insured": "Uninsured", "aware_online": "Unaware of online booking",
             "completed_booking": "Has not booked online", "intent_top2": "Low booking intent", "need_flag": "Unlikely to need package",
             "home_pref": "Prefers lab collection", "tests_num": "Infrequent tester", "pays_self": "Others pay"}


def segment_name(zrow):
    """Name a segment from its two most distinctive features (largest absolute z-score of at least 0.5)."""
    order = zrow.abs().sort_values(ascending=False)
    parts = []
    for feat in order.index:
        if abs(zrow[feat]) < 0.5 or len(parts) == 2:
            break
        parts.append(FEATURES[feat] if zrow[feat] > 0 else NEG_LABEL[feat])
    return " · ".join(parts) if parts else "Middle of the market"
