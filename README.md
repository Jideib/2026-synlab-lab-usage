# SYNLAB Nigeria market-intelligence dashboard

Five-page Streamlit dashboard for the Laboratory Services Survey (Lagos and FCT Abuja, Nov 2026),
styled to the SYNLAB blue-only theme.

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```
`data/synlab_dashboard.csv` and `data/survey_summary.json` are included, so it runs as is.

## Respondent base
| Count | Meaning |
|---|---|
| 1,373 | Submissions received |
| 1,330 | Total respondents after removing 43 duplicate entries (shown on the cover) |
| 959 | Eligible after screening (the base for every page) |
| 940 | Eligible and completed the full questionnaire |

## Rebuild the data files (optional)
```bash
python prepare_data.py \
  --clean SYNLAB_survey_deduplicated_completes.csv \
  --all   SYNLAB_survey_deduplicated.csv \
  --raw   "<original Kobo export>.csv" \
  --out   data/synlab_dashboard.csv
```
* `--all` gives the 1,330 total and `--raw` gives the 1,373 submissions received. No enumerator information is carried into the dashboard data.
* Lab names are grouped into brands by `PROVIDER_RULES` in `prepare_data.py`. Add a line there to group more labs.

## Pages
| Page | Content |
|---|---|
| Home | Cover with total vs eligible respondents, headline KPIs, three insights, navigation |
| 1 Executive Overview | KPI rows, region donut, booking funnel, price-response distribution, 4 takeaways |
| 2 Market & Digital Funnel | Awareness by LGA, funnel drop-off, non-completion reasons, discovery channels, location drill-down, CX radar, 4 insight cards |
| 3 Customer Insights | Filters, age and employment x price heatmaps, insurance vs booking, collection and payment, area cards, 8 insights |
| 4 Competitive & Barriers | Provider share, barriers, abandonment, trust drivers, results channel, SWOT grid, 4 insight cards |
| 5 Strategic Analytics | Region x price heatmap, price vs likelihood, home-fee histogram, tier table, **logistic regression**, **k-means segmentation**, revenue calculator, 4 recommendations |

Navigation sits at the bottom of every page. Every page has a Region filter; page 3 adds LGA, age, employment and insurance filters.

## Strategic Analytics: models
* **Logistic regression** (`analytics.py`, unpenalised, fitted by iteratively reweighted least squares). Pick the outcome: completed an online booking, or
  would book at the price shown. Predictors: region, age, insurance, employment, digital capability, gender (plus price for the second outcome).
  The page shows odds ratios with 95% intervals,
  fit statistics (AUC, McFadden R², accuracy) and a plain-language reading generated from the results.
* **K-means** on ten standardised respondent characteristics, k-means++ starts, 10 restarts. Choose 2 to 6 segments; the default is the
  best silhouette between 3 and 5. Each segment gets a profile, a name from its most distinctive traits and a suggested action.
* Both were checked against scikit-learn (coefficients and AUC agree to six decimals; k-means inertia and silhouette agree). scikit-learn is **not** required to run the dashboard.

## Things to know
* **Price test is stated intent, not willingness to pay.** Each respondent saw one random price (NGN 15,000, 35,000 or 75,000),
  so there is no modal or mean WTP. The "WTP" views show would-book rates by price shown.
* **Small groups.** Heatmap cells, area cards and rankings under 15 respondents are suppressed or marked n/a.
* **Calculator.** The revenue scenario multiplies prospects x need x stated intent x a realisation assumption (default 40%) x price.
  The realisation rate is an assumption, not a survey result.
* The funnel counts attempted and completed bookings among respondents who were aware. Some completed a booking without awareness
  (an HMO or employer booked for them); the page shows that count.
