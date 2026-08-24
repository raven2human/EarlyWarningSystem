# Spend Erosion Early Warning — Streamlit dashboard (v2)
# Reads only cached files in ./dashboard_data (built by prepare_dashboard_data.ipynb v2).
# Nothing is retrained here; the app is a decision console over precomputed results.

import json
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="Spend Erosion Early Warning", page_icon="📉", layout="wide")
DATA = Path(__file__).parent / "dashboard_data"

TEST_SNAPSHOTS = [9, 10, 11]          # temporally held-out era (snapshots 7-8 embargoed)
SEGMENT_NOTES = {
    "seg_2": "⚠️ erosion hotspot — target",
    "seg_6": "⚠️ erosion hotspot — target",
    "seg_1": "✅ zero observed erosion — skip",
    "seg_5": "✅ zero observed erosion — skip",
}

# human-readable segment names, taken from the k-prototypes profile table (notebook §7c)
SEGMENT_NAMES = {
    "seg_0": "Mainstream frequent users",
    "seg_1": "Affluent heavy spenders",
    "seg_2": "Low-engagement mid-age",
    "seg_3": "Young lifestyle spenders",
    "seg_4": "Older mainstream (largest group)",
    "seg_5": "Selective high-value buyers",
    "seg_6": "Disengaged seniors",
}

# plain-language names for the model features (used in the why-flagged and what-if pages)
FEATURE_LABELS = {
    "spend_log": "Spending level (log)",
    "spend_cv": "Spending volatility",
    "spend_trend": "Spending trend",
    "spend_trend_vs_cohort": "Spending trend vs cohort",
    "n_tx_avg": "Transactions per month",
    "n_tx_trend": "Transactions (trend)",
    "active_days_avg": "Active days per month",
    "active_days_trend": "Active days (trend)",
    "n_merchants_avg": "Distinct merchants",
    "n_merchants_trend": "Distinct merchants (trend)",
    "n_categories_avg": "Distinct categories",
    "n_categories_trend": "Distinct categories (trend)",
    "avg_ticket_avg": "Average purchase size",
    "weekend_share_avg": "Weekend spending share",
    "night_share_avg": "Night-time spending share",
    "geo_spread_lat_avg": "Geographic spread (N–S)",
    "geo_spread_long_avg": "Geographic spread (E–W)",
    "disc_share": "Discretionary spending share",
    "disc_share_trend": "Discretionary share (trend)",
    "llm_disc_share": "Discretionary share (LLM-graded)",
    "llm_disc_share_trend": "Discretionary share, LLM (trend)",
    "income_proxy": "Income proxy (from job title)",
    "age": "Age",
}


def pretty_feature(name):
    """readable label for a feature; falls back to a tidied version of the raw name"""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.startswith("share_"):
        return "Category share: " + name[6:].replace("_", " ")
    if name.startswith("occ_"):
        return "Occupation: " + name[4:].replace("_", " ")
    if name.startswith("seg_"):
        return SEGMENT_NAMES.get(name, name)
    return name.replace("_", " ").capitalize()


@st.cache_data
def load_data():
    scores = pd.read_parquet(DATA / "risk_scores.parquet")
    monthly = pd.read_parquet(DATA / "monthly_spend.parquet").set_index("cc_num")
    model_data = pd.read_parquet(DATA / "model_data.parquet")
    results = json.loads((DATA / "results.json").read_text())
    explain = json.loads((DATA / "explain_model.json").read_text())
    return scores, monthly, model_data, results, explain


def risk_from_features(x, explain):
    """logistic risk = sigmoid(intercept + sum(coef * standardized feature))"""
    coef = np.array(explain["coef"])
    mean, std = np.array(explain["mean"]), np.array(explain["std"])
    z = (x - mean) / np.where(std == 0, 1, std)
    linear = explain.get("intercept", 0.0) + float(np.dot(coef, z))
    return 1.0 / (1.0 + np.exp(-linear))


scores, monthly, model_data, results, explain = load_data()
test_scores = scores[scores["t"].isin(TEST_SNAPSHOTS)]
HAS_GEO = "lat" in scores.columns
HAS_INTERCEPT = "intercept" in explain

st.sidebar.title("📉 Spend Erosion EWS")
page = st.sidebar.radio("Page", [
    "Story mode", "Overview", "Campaign simulator", "Cost-benefit", "Risk map",
    "Alerts — risk movers", "Customer drill-down", "What-if simulator", "Segment explorer",
    "Model lab", "Label explorer", "Drift monitor"])
st.sidebar.caption(
    "Master's term project — behavioral early warning for credit-card spend erosion. "
    "All numbers come from cached, temporally held-out evaluation results.")

# short in-app help shown at the top of every page (full guide: DASHBOARD_GUIDE.md in the repo)
PAGE_HELP = {
    "Story mode": "A ten-slide walkthrough of the whole project, from raw transactions to the "
        "retention plan. Use the buttons to step through, or tick Autoplay to let it run.",
    "Overview": "The project's headline numbers: cohort size, test-era erosion rate, best model "
        "performance, and the cohort's monthly spend. Note the 2019 spending ramp — it is why "
        "erosion is measured relative to the cohort median, not in absolute dollars.",
    "Campaign simulator": "The main decision tool. Choose a snapshot and a label definition, then "
        "drag the contact-budget slider: precision, recall, and at-risk value covered update live. "
        "The table below is the actual contact list, downloadable as CSV.",
    "Cost-benefit": "The campaign in money terms. Enter cost per contact, value saved per retained "
        "eroder, and offer success rate; the curve shows net benefit at every budget and marks the "
        "optimum. Change the economics and watch the optimum move.",
    "Risk map": "Each dot is a customer: color = risk (green→red), size = monthly value; hover for "
        "details. Note: geography has NO predictive value in this simulated data — this is an "
        "operational view for campaign planning, not an analytical finding.",
    "Alerts — risk movers": "Early warning in its purest form: customers whose risk score rose most "
        "since the previous snapshot, with their full risk trajectories. A rising score is "
        "actionable before erosion completes.",
    "Customer drill-down": "One customer under the microscope: spend trajectory (observation window "
        "highlighted), risk history, and the 'why flagged' panel — each feature's signed "
        "contribution to the risk score. Red pushes risk up, blue pulls it down.",
    "What-if simulator": "Change a customer's behavior with the sliders and watch the risk score "
        "respond, recomputed live from the model's coefficients. Try cutting spend −30% (risk "
        "rises) or adding a positive spend trend (risk falls).",
    "Segment explorer": "Seven behavioral segments from k-prototypes clustering (k chosen by "
        "experiment). Erosion concentrates in two of them, and the model's risk scores agree — "
        "two affluent segments show zero erosion and need no retention budget.",
    "Model lab": "The evidence room: the nine-model benchmark (TabPFN leads), the ablation study "
        "(what each feature family added per model), and the raw-lag time-series baseline "
        "(feature engineering beats raw history).",
    "Label explorer": "How the erosion definition was chosen: slide through the five candidate "
        "thresholds and see the four quality criteria plus the verdict for each. The label was "
        "engineered by experiment, not picked by taste.",
    "Drift monitor": "Erosion rate per snapshot, split into train / embargoed / test eras. The "
        "prevalence drift (8.6% → 2.5%) is explained here — in production this page would drive "
        "quarterly threshold recalibration.",
}
with st.expander("ℹ️ What does this page show?"):
    st.markdown(PAGE_HELP.get(page, ""))


# -------------------------------------------------------------- Story mode
if page == "Story mode":
    st.title("Story mode — the project in ten slides")
    FIGDIR = Path(__file__).parent / "figures"

    slides = [
        ("project_pipeline.png", "What we built",
         "1.3M transactions become a monthly ranking of customers at risk of spending erosion."),
        ("distributions.png", "The data is noisy",
         "A typical customer's monthly spend swings ±38%, so single months mean nothing."),
        ("spend_decomposition.png", "Everyone accelerates",
         "Transactions per customer rose 42% in 2019 — decline must be measured against the cohort."),
        ("persistence.png", "The future is predictable",
         "Past 3-month spend predicts future 3-month spend at r = 0.86."),
        ("anim_sliding_window.gif", "How we build examples",
         "A 3+3 month window slides across time; snapshots 7-8 are dropped to avoid overlap."),
        ("label_quality.png", "The label was tested",
         "Erosion labels repeat 4.9x more often than chance — a real state, not noise."),
        ("segment_heatmap.png", "Seven customer segments",
         "Named groups from k-prototypes, with k chosen by experiment rather than by taste."),
        ("zoo_curves.png", "Nine models compared",
         "TabPFN leads; at 2.5% positives the precision-recall view is the honest one."),
        ("anim_budget_sweep.gif", "Choosing a budget",
         "As the contact list grows, recall rises and precision falls — pick your operating point."),
        ("segment_risk_overview.png", "Where to spend the budget",
         "Two segments carry the risk; two affluent segments need no retention spend at all."),
    ]

    if "slide" not in st.session_state:
        st.session_state.slide = 0

    c1, c2, c3, c4 = st.columns([1, 1, 2, 4])
    if c1.button("← Prev"):
        st.session_state.slide = (st.session_state.slide - 1) % len(slides)
    if c2.button("Next →"):
        st.session_state.slide = (st.session_state.slide + 1) % len(slides)
    autoplay = c3.checkbox("Autoplay")

    i = st.session_state.slide
    filename, headline, narration = slides[i]
    st.subheader(f"{i + 1}. {headline}")
    st.write(narration)

    image_path = FIGDIR / filename
    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
    else:
        st.warning(f"Figure not found: {filename} — copy it into the figures/ folder next to app.py.")
    st.progress((i + 1) / len(slides))

    if autoplay:
        time.sleep(6)
        st.session_state.slide = (i + 1) % len(slides)
        st.rerun()


# ---------------------------------------------------------------- Overview
elif page == "Overview":
    st.title("Behavioral Early Warning System for Credit Card Spend Erosion")
    st.write(
        "We predict whether a customer's spending will erode **relative to the cohort** in the "
        "next 3 months, using the previous 3 months of behavior. Explore the pages on the left.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Behavioral customers", f"{scores['cc_num'].nunique()}")
    c2.metric("Test-era erosion rate", f"{test_scores['erosion_25'].mean():.1%}")
    best = max(results["model_zoo"], key=lambda r: r["PR_AUC"])
    c3.metric(f"Best model ({best['model']})", f"PR-AUC {best['PR_AUC']:.3f}",
              f"{best['PR_AUC'] / test_scores['erosion_25'].mean():.1f}× no-skill")
    c4.metric("Recall @ top-10% budget", f"{best['recall@10%']:.0%}")

    st.subheader("Cohort monthly spend")
    total = monthly.sum(axis=0).rename_axis("month").reset_index(name="total_spend")
    st.altair_chart(
        alt.Chart(total).mark_bar(color="#4C78A8").encode(
            x=alt.X("month:N", sort=None), y="total_spend:Q",
            tooltip=["month", alt.Tooltip("total_spend:Q", format=",.0f")]),
        use_container_width=True)
    st.caption(
        "Total spend roughly doubles during 2019 — a uniform frequency ramp in the simulation. "
        "This is why erosion is measured relative to the cohort median, not in absolute dollars.")


# ---------------------------------------------------- Campaign simulator
elif page == "Campaign simulator":
    st.title("Campaign simulator")
    st.write(
        "A retention team has a **contact budget**, not a probability threshold. "
        "Slide the budget and watch the trade-off between precision and recall.")

    col_a, col_b, col_c = st.columns(3)
    snap = col_a.selectbox("Test snapshot", TEST_SNAPSHOTS, index=2)
    label_options = [c for c in ["erosion_20", "erosion_25", "erosion_30"] if c in scores.columns]
    label = col_b.selectbox("Label definition (τ what-if)", label_options,
                            index=label_options.index("erosion_25"))
    budget = col_c.slider("Contact budget (% of customers)", 1, 30, 10)

    view = scores[scores["t"] == snap].sort_values("risk", ascending=False).reset_index(drop=True)
    k = max(1, int(len(view) * budget / 100))
    contacted = view.head(k)
    precision = contacted[label].mean()
    recall = contacted[label].sum() / max(view[label].sum(), 1)
    base = view[label].mean()
    value_covered = (contacted.loc[contacted[label] == 1, "monthly_value"].sum()
                     / max(view.loc[view[label] == 1, "monthly_value"].sum(), 1))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers contacted", f"{k}")
    c2.metric("Precision", f"{precision:.1%}", f"{precision / max(base, 1e-9):.1f}× random")
    c3.metric("Recall (eroders caught)", f"{recall:.1%}")
    c4.metric("At-risk value covered", f"{value_covered:.1%}")
    c5.metric("Base erosion rate", f"{base:.1%}")

    view["decile"] = pd.qcut(view["risk"], 10, labels=False, duplicates="drop")
    lift = view.groupby("decile")[label].mean().reset_index()
    lift["decile"] = 9 - lift["decile"]  # 0 = highest risk
    chart = alt.Chart(lift).mark_bar(color="#E45756").encode(
        x=alt.X("decile:O", title="risk decile (0 = highest risk)"),
        y=alt.Y(f"{label}:Q", title="observed erosion rate", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip(f"{label}:Q", format=".1%")])
    rule = alt.Chart(pd.DataFrame({"y": [base]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    st.altair_chart(chart + rule, use_container_width=True)

    st.subheader("Contact list")
    list_cols = ["cc_num", "risk", "segment", "monthly_value", label]
    if HAS_GEO:
        list_cols.append("state")
    st.dataframe(contacted.head(25)[list_cols].rename(columns={label: "actually_eroded"}),
                 use_container_width=True)
    st.download_button(
        "⬇️ Download full contact list (CSV)",
        contacted[list_cols].to_csv(index=False).encode(),
        file_name=f"contact_list_snapshot{snap}_top{budget}pct.csv",
        mime="text/csv")
    st.caption("Recommended policy: high-touch offers for the top 5%, automated incentives for the next 15%.")


# ---------------------------------------------------------- Cost-benefit
elif page == "Cost-benefit":
    st.title("Cost–benefit: where is the optimal budget?")
    st.write(
        "Turn the campaign frontier into money. Net benefit(k) = "
        "**success rate × value saved × true positives caught − cost × customers contacted**.")

    c1, c2, c3, c4 = st.columns(4)
    snap = c1.selectbox("Test snapshot", TEST_SNAPSHOTS, index=2)
    cost = c2.number_input("Cost per contact ($)", 1, 500, 10)
    value = c3.number_input("Value saved per retained eroder ($)", 10, 20000, 500)
    success = c4.slider("Offer success rate", 0.05, 1.0, 0.30)

    view = scores[scores["t"] == snap].sort_values("risk", ascending=False).reset_index(drop=True)
    rows = []
    for pct in range(1, 31):
        k = max(1, int(len(view) * pct / 100))
        tp = int(view.head(k)["erosion_25"].sum())
        rows.append({"budget_pct": pct, "contacted": k, "true_positives": tp,
                     "net_benefit": success * value * tp - cost * k})
    curve = pd.DataFrame(rows)
    best_row = curve.loc[curve["net_benefit"].idxmax()]

    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal budget", f"{int(best_row['budget_pct'])}% "
              f"({int(best_row['contacted'])} customers)")
    c2.metric("Net benefit at optimum", f"${best_row['net_benefit']:,.0f}")
    c3.metric("Eroders caught at optimum", f"{int(best_row['true_positives'])}")

    line = alt.Chart(curve).mark_line(point=True, color="#4C78A8").encode(
        x=alt.X("budget_pct:Q", title="contact budget (%)"),
        y=alt.Y("net_benefit:Q", title="net benefit ($)"),
        tooltip=["budget_pct", "contacted", "true_positives",
                 alt.Tooltip("net_benefit:Q", format=",.0f")])
    optimum = alt.Chart(pd.DataFrame([best_row])).mark_point(
        size=150, color="#E45756", filled=True).encode(x="budget_pct:Q", y="net_benefit:Q")
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    st.altair_chart(line + optimum + zero, use_container_width=True)
    st.caption(
        "The curve rises while marginal contacts are precise, and falls once the ranking runs out "
        "of true eroders. Change the economics and watch the optimum move — expensive offers push "
        "it left (contact fewer, more precisely), high saved value pushes it right.")


# -------------------------------------------------------------- Risk map
elif page == "Risk map":
    st.title("Risk map — where are the at-risk customers?")
    if not HAS_GEO:
        st.error("Location columns missing — regenerate dashboard_data with "
                 "prepare_dashboard_data.ipynb v2 and replace the folder in the repo.")
    else:
        snap = st.selectbox("Test snapshot", TEST_SNAPSHOTS, index=2)
        geo = scores[(scores["t"] == snap) & scores["lat"].notna()].copy()
        # color by risk percentile so the palette spreads even when risks cluster
        pct = geo["risk"].rank(pct=True)
        geo["r"] = (255 * pct).astype(int)
        geo["g"] = (255 * (1 - pct)).astype(int)
        geo["radius"] = 8000 + 40 * np.sqrt(geo["monthly_value"])
        geo["risk_pct"] = (geo["risk"] * 100).round(1)

        layer = pdk.Layer(
            "ScatterplotLayer", data=geo,
            get_position="[lon, lat]", get_radius="radius",
            get_fill_color="[r, g, 60, 160]", pickable=True)
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=39.5, longitude=-98.3, zoom=3.3),
            tooltip={"text": "customer {cc_num}\nrisk: {risk_pct}%\nsegment: {segment}\n"
                             "monthly value: ${monthly_value}\nstate: {state}"},
            map_style=None)
        st.pydeck_chart(deck)
        st.caption(
            "Hover a dot: red = high erosion risk, green = low; dot size scales with monthly value. "
            "Note: our analysis found geography has NO predictive value in this simulated data — "
            "this is an operational view (regional campaign planning), not an analytical finding.")

        st.subheader("Mean risk by state (top 15)")
        by_state = (geo.groupby("state").agg(customers=("risk", "size"), mean_risk=("risk", "mean"))
                    .sort_values("mean_risk", ascending=False).head(15).round(3))
        st.dataframe(by_state, use_container_width=True)


# ------------------------------------------------- Alerts (risk movers)
elif page == "Alerts — risk movers":
    st.title("Alerts — whose risk is rising fastest?")
    st.write(
        "The purest form of *early warning*: customers whose risk score jumped most since the "
        "previous snapshot. A rising score is actionable before the erosion completes.")

    snap = st.selectbox("Snapshot", TEST_SNAPSHOTS, index=2)
    now = scores[scores["t"] == snap][["cc_num", "risk", "segment", "monthly_value", "erosion_25"]]
    before = scores[scores["t"] == snap - 1][["cc_num", "risk"]].rename(columns={"risk": "risk_prev"})
    movers = now.merge(before, on="cc_num")
    movers["risk_change"] = movers["risk"] - movers["risk_prev"]
    movers = movers.sort_values("risk_change", ascending=False)

    c1, c2 = st.columns(2)
    c1.metric("Customers with rising risk", f"{(movers['risk_change'] > 0).sum()}")
    c2.metric("Biggest single jump", f"+{movers['risk_change'].max():.2f}")

    st.subheader("Top 20 risers")
    st.dataframe(movers.head(20).round(3), use_container_width=True)

    top_ids = movers.head(20)["cc_num"].tolist()
    hist = scores[scores["cc_num"].isin(top_ids)][["cc_num", "t", "risk"]]
    st.altair_chart(
        alt.Chart(hist).mark_line(opacity=0.6).encode(
            x="t:O", y="risk:Q", color=alt.Color("cc_num:N", legend=None),
            tooltip=["cc_num", "t", alt.Tooltip("risk:Q", format=".2f")]),
        use_container_width=True)
    st.caption("Risk trajectories of the top-20 risers across all snapshots.")


# --------------------------------------------------- Customer drill-down
elif page == "Customer drill-down":
    st.title("Customer drill-down")
    snap = st.selectbox("Snapshot", TEST_SNAPSHOTS, index=2)
    snap_scores = scores[scores["t"] == snap].sort_values("risk", ascending=False)
    pick = st.selectbox("Customer (sorted by risk, top 50 shown)",
                        snap_scores["cc_num"].head(50).tolist())
    cust = snap_scores[snap_scores["cc_num"] == pick].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk score", f"{cust['risk']:.2f}")
    c2.metric("Segment", SEGMENT_NAMES.get(cust["segment"], str(cust["segment"])),
              SEGMENT_NOTES.get(cust["segment"], "monitor"))
    c3.metric("Monthly value", f"${cust['monthly_value']:,.0f}")
    c4.metric("Actually eroded (label)", "yes" if cust["erosion_25"] == 1 else "no")

    left, right = st.columns(2)
    with left:
        st.subheader("Monthly spend trajectory")
        traj = monthly.loc[pick].rename_axis("month").reset_index(name="spend")
        obs_months = list(monthly.columns[snap:snap + 3])
        traj["window"] = np.where(traj["month"].isin(obs_months), "observation", "history")
        st.altair_chart(
            alt.Chart(traj).mark_bar().encode(
                x=alt.X("month:N", sort=None), y="spend:Q",
                color=alt.Color("window:N", scale=alt.Scale(
                    domain=["history", "observation"], range=["#B0BEC5", "#4C78A8"])),
                tooltip=["month", alt.Tooltip("spend:Q", format=",.0f")]),
            use_container_width=True)
    with right:
        st.subheader("Risk history across snapshots")
        hist = scores[scores["cc_num"] == pick][["t", "risk", "erosion_25"]]
        st.altair_chart(
            alt.Chart(hist).mark_line(point=True, color="#E45756").encode(
                x="t:O", y=alt.Y("risk:Q", scale=alt.Scale(domain=[0, 1])),
                tooltip=["t", alt.Tooltip("risk:Q", format=".2f"), "erosion_25"]),
            use_container_width=True)

    st.subheader("Why is this customer flagged?")
    st.caption("Contribution of each feature to the (logistic) risk score: "
               "coefficient × standardized feature value. Positive pushes risk up.")
    row = model_data[(model_data["cc_num"] == pick) & (model_data["t"] == snap)]
    if len(row):
        feats, coef = explain["features"], np.array(explain["coef"])
        mean, std = np.array(explain["mean"]), np.array(explain["std"])
        x = row[feats].to_numpy(dtype=float)[0]
        contrib = coef * (x - mean) / np.where(std == 0, 1, std)
        top = (pd.DataFrame({"feature": [pretty_feature(f) for f in feats],
                             "contribution": contrib})
               .reindex(np.abs(contrib).argsort()[::-1][:12]))
        st.altair_chart(
            alt.Chart(top).mark_bar().encode(
                x="contribution:Q", y=alt.Y("feature:N", sort="-x"),
                color=alt.condition(alt.datum.contribution > 0,
                                    alt.value("#E45756"), alt.value("#4C78A8")),
                tooltip=["feature", alt.Tooltip("contribution:Q", format=".2f")]),
            use_container_width=True)


# ------------------------------------------------------ What-if simulator
elif page == "What-if simulator":
    st.title("What-if simulator")
    st.write(
        "Change a customer's behavior with the sliders and watch the risk score respond. "
        "The score is recomputed live from the logistic model's coefficients — no server, no retraining.")
    if not HAS_INTERCEPT:
        st.error("Model intercept missing — regenerate dashboard_data with "
                 "prepare_dashboard_data.ipynb v2 and replace the folder in the repo.")
    else:
        snap = st.selectbox("Snapshot", TEST_SNAPSHOTS, index=2)
        snap_scores = scores[scores["t"] == snap].sort_values("risk", ascending=False)
        pick = st.selectbox("Customer", snap_scores["cc_num"].head(50).tolist())
        row = model_data[(model_data["cc_num"] == pick) & (model_data["t"] == snap)]
        feats = explain["features"]
        x0 = row[feats].to_numpy(dtype=float)[0]
        base_risk = risk_from_features(x0, explain)

        st.subheader("Adjust behavior")
        s1, s2, s3 = st.columns(3)
        spend_pct = s1.slider("Total spend change (%)", -50, 50, 0)
        tx_pct = s2.slider("Transaction count change (%)", -50, 50, 0)
        merch_pct = s3.slider("Distinct merchants change (%)", -50, 50, 0)
        s4, s5, s6 = st.columns(3)
        disc_pp = s4.slider("Discretionary share change (pp)", -20, 20, 0)
        trend_delta = s5.slider("Spend trend change", -0.5, 0.5, 0.0, 0.05)
        days_pct = s6.slider("Active days change (%)", -50, 50, 0)

        x = x0.copy()
        idx = {f: i for i, f in enumerate(feats)}
        if "spend_log" in idx:
            x[idx["spend_log"]] = np.log1p(np.expm1(x0[idx["spend_log"]]) * (1 + spend_pct / 100))
        for col, pct in [("n_tx_avg", tx_pct), ("n_merchants_avg", merch_pct),
                         ("active_days_avg", days_pct)]:
            if col in idx:
                x[idx[col]] = x0[idx[col]] * (1 + pct / 100)
        for col in ["disc_share", "llm_disc_share"]:
            if col in idx:
                x[idx[col]] = float(np.clip(x0[idx[col]] + disc_pp / 100, 0, 1))
        for col in ["spend_trend", "spend_trend_vs_cohort"]:
            if col in idx:
                x[idx[col]] = x0[idx[col]] + trend_delta
        new_risk = risk_from_features(x, explain)

        c1, c2, c3 = st.columns(3)
        c1.metric("Original risk", f"{base_risk:.2f}")
        c2.metric("What-if risk", f"{new_risk:.2f}", f"{new_risk - base_risk:+.2f}")
        c3.metric("Interpretation", "⚠️ riskier" if new_risk > base_risk + 0.01
                  else ("✅ safer" if new_risk < base_risk - 0.01 else "≈ unchanged"))
        st.caption(
            "Try it: cut spend −30% and watch risk rise; add a positive spend trend and watch it fall. "
            "This is the model's logic made tangible — the same coefficients as the "
            "'why flagged' panel, applied to hypothetical behavior.")


# ----------------------------------------------------- Segment explorer
elif page == "Segment explorer":
    st.title("Segment explorer")
    st.write("Seven behavioral segments from k-prototypes clustering "
             "(k chosen by inner-validation experiment). Erosion concentrates in two of them.")

    seg = results["erosion_by_segment"]
    seg_df = pd.DataFrame({
        "segment": list(seg["n"].keys()),
        "customers (test rows)": list(seg["n"].values()),
        "erosion_rate": list(seg["erosion_rate"].values()),
        "mean_model_risk": [round(v, 3) for v in seg["mean_risk"].values()],
    })
    seg_df["name"] = seg_df["segment"].map(SEGMENT_NAMES).fillna("")
    seg_df["recommendation"] = seg_df["segment"].map(SEGMENT_NOTES).fillna("monitor")
    seg_df = seg_df[["segment", "name", "customers (test rows)", "erosion_rate",
                     "mean_model_risk", "recommendation"]]

    base_rate = test_scores["erosion_25"].mean()
    chart = alt.Chart(seg_df).mark_bar().encode(
        x=alt.X("segment:N"),
        y=alt.Y("erosion_rate:Q", axis=alt.Axis(format="%")),
        color=alt.condition(alt.datum.erosion_rate > base_rate,
                            alt.value("#E45756"), alt.value("#4C78A8")),
        tooltip=["segment", alt.Tooltip("erosion_rate:Q", format=".1%"), "mean_model_risk"])
    rule = alt.Chart(pd.DataFrame({"y": [base_rate]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    st.altair_chart(chart + rule, use_container_width=True)
    st.dataframe(seg_df, use_container_width=True)

    st.subheader("Behavioral profile per segment (test-era means)")
    profile_cols = ["spend_log", "n_tx_avg", "avg_ticket_avg", "disc_share", "night_share_avg", "age"]
    have = [c for c in profile_cols if c in model_data.columns]
    seg_cols = [c for c in model_data.columns if c.startswith("seg_")]
    md_test = model_data[model_data["t"].isin(TEST_SNAPSHOTS)].copy()
    md_test["segment"] = md_test[seg_cols].idxmax(axis=1)
    st.dataframe(md_test.groupby("segment")[have].mean().round(2), use_container_width=True)
    st.caption("The model's mean risk is highest for exactly the two segments with the highest "
               "observed erosion — the unsupervised segmentation and the supervised model agree.")


# ------------------------------------------------------------ Model lab
elif page == "Model lab":
    st.title("Model lab")

    st.subheader("Nine-model benchmark (full feature set)")
    zoo = pd.DataFrame(results["model_zoo"])
    st.altair_chart(
        alt.Chart(zoo).mark_bar().encode(
            x=alt.X("PR_AUC:Q"), y=alt.Y("model:N", sort="-x"),
            color=alt.condition(alt.datum.model == "TabPFN",
                                alt.value("#E45756"), alt.value("#4C78A8")),
            tooltip=["model", "PR_AUC", "ROC_AUC"]),
        use_container_width=True)
    st.caption(f"No-skill PR baseline = test prevalence = {test_scores['erosion_25'].mean():.3f}. "
               "TabPFN (a tabular foundation model doing in-context learning) leads; "
               "Gaussian Naive Bayes trails — its independence assumption is violated by design.")
    st.dataframe(zoo, use_container_width=True)

    st.subheader("Ablation study (feature families added one at a time)")
    st.dataframe(pd.DataFrame(results["ablation"]), use_container_width=True)
    st.caption("Segments lift the linear model most (+0.032 PR); LLM features add little "
               "(they correlate 0.92 with the manual mapping); XGBoost overfits wider feature sets.")

    st.subheader("Raw-lag (time series) baseline")
    st.dataframe(pd.DataFrame(results["ar_baseline"]), use_container_width=True)
    st.caption("Raw lags alone approach no-skill (AR-3); with 6 lags only the linear model recovers "
               "(a weighted difference of log-lags is a growth rate). Engineered features win. "
               "Per-customer lag-1 autocorrelation is ~0: the signal is between customers, "
               "not in monthly wiggles.")


# -------------------------------------------------------- Label explorer
elif page == "Label explorer":
    st.title("Label explorer — how the erosion definition was chosen")
    st.write(
        "Erosion = customer's 3-month-ahead spend change minus the **cohort median** change, "
        "below a threshold −τ. Five thresholds were auditioned against four pre-declared criteria.")

    q = results["labels"]["quality_table"]
    taus = list(q["prevalence"].keys())
    tau = st.select_slider("Threshold τ", options=taus, value="0.25")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prevalence", f"{q['prevalence'][tau]:.1%}")
    c2.metric("Winsor/raw agreement", f"{q['agreement'][tau]:.1%}")
    c3.metric("Jaccard of positives", f"{q['jaccard_of_positives'][tau]:.2f}")
    c4.metric("Persistence lift", f"{q['persistence_lift'][tau]:.1f}×")

    verdicts = {
        "0.15": "❌ Rejected: sits at the ~22% noise floor of 3-month averages; weakest persistence.",
        "0.2": "🟡 Kept as sensitivity cutoff (11.3% prevalence).",
        "0.25": "✅ CHOSEN: prevalence in the predicted 3–10% band, robust to winsorization, "
                "and labeled customers are ~5× likelier than baseline to erode again — a state, not noise.",
        "0.3": "🟡 Kept as sensitivity cutoff (4.2% prevalence).",
        "0.4": "❌ Rejected: zero-positive snapshots (untrainable) and repeat probability collapses "
               "— captures one-off shocks, not disengagement.",
    }
    st.info(verdicts.get(tau, ""))

    st.subheader("All criteria, all thresholds")
    st.dataframe(pd.DataFrame(q).rename_axis("τ"), use_container_width=True)
    st.subheader("Alternative window shapes (prevalence)")
    st.dataframe(pd.DataFrame(results["labels"]["config_table"]).rename_axis("τ"),
                 use_container_width=True)
    st.caption("Shorter windows (2+2, 3+1) inflate prevalence at every τ — less averaging means "
               "more noise crossings. 3+3 retained.")


# --------------------------------------------------------- Drift monitor
elif page == "Drift monitor":
    st.title("Drift monitor")
    st.write("If this system ran in production, this page is what an ML engineer would watch.")

    prev = scores.groupby("t")["erosion_25"].mean().reset_index()
    prev["era"] = np.where(prev["t"] <= 6, "train",
                           np.where(prev["t"].isin(TEST_SNAPSHOTS), "test", "embargoed"))
    st.altair_chart(
        alt.Chart(prev).mark_bar().encode(
            x="t:O", y=alt.Y("erosion_25:Q", axis=alt.Axis(format="%"), title="erosion rate"),
            color=alt.Color("era:N", scale=alt.Scale(
                domain=["train", "embargoed", "test"],
                range=["#4C78A8", "#B0BEC5", "#E45756"])),
            tooltip=["t", alt.Tooltip("erosion_25:Q", format=".1%"), "era"]),
        use_container_width=True)

    tr = scores[scores["t"] <= 6]["erosion_25"].mean()
    te = test_scores["erosion_25"].mean()
    c1, c2 = st.columns(2)
    c1.metric("Train-era prevalence", f"{tr:.1%}")
    c2.metric("Test-era prevalence", f"{te:.1%}", f"{(te - tr):+.1%} drift")
    st.warning(
        "Label prevalence drifts from 8.6% to 2.5% between eras: the simulator's frequency ramp "
        "stabilizes in 2020, so fewer customers cross the relative threshold. Ranking metrics "
        "remain valid under this shift, but PR-AUC must always be compared to its own era's "
        "baseline — and production deployment would require quarterly threshold recalibration.")
