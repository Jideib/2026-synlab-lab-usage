# SYNLAB Nigeria Laboratory Services Survey: executive dashboard

Run locally:

    pip install -r requirements.txt
    streamlit run app.py

Deploy on Streamlit Community Cloud: push this folder to a GitHub repository and set the main file to `app.py`.

- `data/SYNLAB_survey_deduplicated_completes.csv`: 959 eligible completes (semicolon-separated); 940 are online-ready.
- Pages: Cover, Overview, Brand Health, Customer Insights, Competitive Intelligence, Strategic Analytics.
- Sidebar filters (city, age) apply to every page except the models, which use all respondents who could judge the package (895) so results match the report.
- Models: logistic regression (booking intent), k-means (segments), price response by city. Stated intent, unweighted sample.
