# Spend Erosion Early Warning — Dashboard

Interactive dashboard for our master's term project: *Behavioral Early Warning System for
Credit Card Spend Erosion*. Predicts which customers' spending will erode relative to the
cohort in the next 3 months, and turns the scores into retention decisions.

**Pages:** Overview · Campaign simulator (budget slider → precision/recall/value) ·
Customer drill-down (trajectory, risk history, why-flagged) · Segment explorer ·
Model lab (9-model benchmark, ablations, time-series baseline) · Label explorer ·
Drift monitor.

All results come from cached, temporally held-out evaluation — the app retrains nothing.

## Deploy (GitHub → Streamlit Community Cloud)

1. **Generate the data files** (once): run `prepare_dashboard_data.ipynb` in Google Colab.
   It writes a `dashboard_data/` folder (~5 MB) to your Drive's `TermProject` folder.
2. **Repo layout** — create a GitHub repo containing:
   ```
   app.py
   requirements.txt
   README.md
   dashboard_data/
       model_data.parquet
       monthly_spend.parquet
       risk_scores.parquet
       explain_model.json
       results.json
   ```
   (Download `dashboard_data/` from Drive and commit it. The raw 338 MB CSV must NOT go
   in the repo — it isn't needed.)
3. **Deploy**: go to https://share.streamlit.io → "Create app" → pick your repo,
   branch `main`, main file `app.py` → Deploy. First build takes ~2 minutes.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Risk scores are produced by the reproducible CPU model (logistic regression, full
  feature set). To swap in TabPFN scores, regenerate `risk_scores.parquet` with the
  `risk` column replaced — the app is agnostic to where scores come from.
- Data is the public simulated Sparkov credit-card dataset; see the project report for
  the full methodology and limitations.
