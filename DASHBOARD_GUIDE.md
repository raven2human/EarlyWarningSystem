# Dashboard Guide — what each page shows and how to use it

This is the user guide for our Streamlit dashboard (Spend Erosion Early Warning System).
The app has 11 pages, selected from the sidebar. Every number in the app comes from
cached, temporally held-out evaluation results — the app never retrains anything.
Risk scores come from the reproducible logistic regression model (full feature set).

---

## 1. Overview
**What it shows:** The headline numbers of the whole project as metric cards — 908
behavioral customers, the test-era erosion rate (~2.5%), the best model's PR-AUC with
its lift over random, and the recall at a 10% contact budget. Below them, a bar chart
of the cohort's total monthly spend.

**What to notice:** Total spending roughly doubles during 2019. That is the reason our
erosion label is measured *relative to the cohort median*, not in absolute dollars —
if everyone accelerates, an absolute threshold would never flag anyone.

**When demoing:** Start here. It answers "what is this and does it work?" in 10 seconds.

---

## 2. Campaign simulator
**What it shows:** The main decision tool. Pick a test snapshot, pick a label
definition (τ = 0.20 / 0.25 / 0.30 — the what-if on our sensitivity analysis), and
drag the contact-budget slider (1–30%). The metrics update live: customers contacted,
precision (and its lift vs random), recall, and the share of at-risk *dollars* covered.
Below: the risk-decile chart (observed erosion rate per predicted-risk decile) and the
actual contact list, downloadable as CSV.

**What to notice:** The precision/recall trade-off is physical here — drag the slider
and watch precision fall as recall rises. At 5% budget roughly 1 in 5 contacted
customers truly erodes (8× better than random); at 20% budget you catch ~80% of all
eroders.

**When demoing:** This is the money slide. Drag the slider slowly while narrating the
trade-off; end on the download button ("and this is the list the retention team gets").

---

## 3. Cost–benefit
**What it shows:** The campaign turned into money. Three inputs: cost per contact,
value saved per retained eroder, and the offer success rate. The app computes
net benefit = success × value × true positives − cost × contacted, for every budget
from 1% to 30%, and marks the optimal budget on the curve.

**What to notice:** The curve rises while the ranking is precise and falls once it
runs out of true eroders. Change the economics and the optimum moves: expensive
offers push it left (contact fewer, more precisely), high saved value pushes it right.

**When demoing:** Ask the audience for a cost and a value, type them in live, and let
the optimum answer. It turns "is 15% PR-AUC good?" into "here is where you make money."

---

## 4. Risk map
**What it shows:** All customers of a test snapshot on a US map. Dot color = risk
percentile (green → red), dot size = the customer's monthly spending value. Hovering a
dot shows the customer ID, risk %, segment, monthly value, and state. Below the map,
a mean-risk-by-state table.

**Honest caveat (also printed on the page):** our analysis found geography has NO
predictive value in this simulated data. The map is an *operational* view (where are
the at-risk customers, for regional campaign planning), not an analytical finding.

**When demoing:** Great visual moment — hover two or three dots — but say the caveat
out loud; it earns credibility.

---

## 5. Alerts — risk movers
**What it shows:** The purest form of "early warning": customers whose risk score
jumped the most since the previous snapshot, as a top-20 table plus their full risk
trajectories over time.

**What to notice:** A *rising* score is actionable before erosion completes — this
page is what a monthly monitoring routine would actually look at first.

---

## 6. Customer drill-down
**What it shows:** One customer under the microscope. Their metric cards (risk,
segment with a target/skip note, monthly value, and whether they actually eroded),
their 17-month spend trajectory with the observation window highlighted in blue,
their risk history across all snapshots, and the "Why is this customer flagged?"
panel — the signed contribution of each feature to their logistic risk score
(red bars push risk up, blue bars pull it down).

**What to notice:** The why-flagged panel is the interpretability story: the model
is not a black box, we can name the reasons for every flag (e.g., "transactions
trending down, discretionary share falling").

**When demoing:** Pick a high-risk customer, read their top two red bars aloud as a
sentence: "flagged because their transaction count is falling and their spend trend
is below the cohort."

---

## 7. What-if simulator
**What it shows:** Pick a customer and change their behavior with six sliders
(total spend %, transaction count %, distinct merchants %, discretionary share,
spend trend, active days %). The risk score is recomputed live from the model's
coefficients — no server, no retraining.

**What to notice:** Cut spend −30% and risk climbs; give the customer a positive
spend trend and risk falls. Because logistic regression is linear in its features,
the *direction* of each slider is the same for every customer — only the size of the
change differs (biggest near risk 0.5, small at the extremes).

**When demoing:** The wow moment. One sentence to say: "this is the model's logic
made touchable — the same coefficients as the why-flagged panel, applied to
hypothetical behavior."

---

## 8. Segment explorer
**What it shows:** The seven k-prototypes segments: erosion rate per segment (bars,
with the base rate as a dashed line), the model's mean risk per segment, a
recommendation chip per segment (target / monitor / skip), and the behavioral profile
table (spend, frequency, ticket, discretionary share, age).

**What to notice:** Erosion concentrates in exactly two segments (~7% rate vs 0–3%
elsewhere), and the model's mean risk is highest for exactly those two — the
unsupervised clustering and the supervised model agree independently. Two affluent
segments show zero erosion: no retention budget should go there.

---

## 9. Model lab
**What it shows:** The evidence room. The nine-model benchmark (PR-AUC bars with
TabPFN highlighted, plus the full metrics table), the ablation study table (what each
feature family added, per model), and the raw-lag time-series baseline table.

**What to notice:** Three honest findings are written as captions: segments lift the
linear model most; LLM features add little (they correlate 0.92 with a manual
mapping); and raw lags alone approach no-skill — feature engineering earned its keep.

**When demoing:** For a technical audience only; skip for business audiences.

---

## 10. Label explorer
**What it shows:** How the erosion definition was chosen. A τ slider walks through
the five candidate thresholds; for each one the four quality criteria update
(prevalence, winsorized/raw agreement, Jaccard overlap, persistence lift) plus a
verdict box explaining why that τ was chosen, kept as a sensitivity cutoff, or
rejected. Below: the full criteria table and the window-shape comparison (2+2 and
3+1 rejected).

**What to notice:** The label was *engineered by experiment*, not picked by taste —
this page is the proof, made interactive.

---

## 11. Drift monitor
**What it shows:** The erosion rate per snapshot, color-coded train / embargoed /
test. Metric cards show train-era prevalence (8.6%) vs test-era (2.5%) and the drift
between them, with a warning box explaining the cause (the simulator's growth ramp
stabilizes in 2020) and the production implication (quarterly threshold
recalibration).

**What to notice:** This is the "if this ran in production" page — it shows we thought
about what happens *after* deployment, not just the model score.
