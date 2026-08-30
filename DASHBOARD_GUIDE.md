# Dashboard Guide — Behavioral Early Warning System for Credit Card Spend Erosion

**Team:** Gizem Baştürk Yeşilova · Seda Ulutaş · Şafak Erkaya
**Purpose of this document:** what every screen shows, how to read it, what to say out loud, and what to answer when questioned.

---

## 1. What this application is

A **decision console**, not a modelling environment. Every number it displays was computed in `TermProject_Final.ipynb` and cached to `dashboard_data/`. The app retrains nothing and recomputes nothing except two live calculations that are explicitly labelled (the what-if risk score and the cost-benefit curve).

This matters, and it is worth saying in the first thirty seconds: **the dashboard cannot accidentally produce a number that differs from the report.** It reads the same five files the report was written from.

| file | contents |
|---|---|
| `risk_scores.parquet` | one row per (customer, snapshot): risk score, actual label, monthly value |
| `monthly_spend.parquet` | 908 × 17 spend grid, for trajectory charts |
| `model_data.parquet` | the 55-feature modelling table |
| `results.json` | every evaluation result — ablation, model zoo, label criteria, campaign frontier |
| `explain_model.json` | logistic coefficients, means, standard deviations (drives "why flagged" and what-if) |

**Sidebar navigation:** twelve pages. Story mode is the presentation; the other eleven are the evidence and the tools.

**Every page has an `ℹ️ What does this page show?` expander** at the top. If you blank during the viva, open it — the answer is already written.

---

## 2. Story mode — the fourteen slides

Story mode is the walkthrough: raw transactions to retention policy in fourteen steps. Each slide has three fields, and they are deliberately different registers:

- **Objective** — what we were trying to establish at that stage
- **Findings** — what the evidence actually said
- **Implication** — the "so what", written for you to speak aloud

**Controls:** *Previous* / *Next* to move at your own pace. Animated slides offer a **Playback** control: *Step through manually* gives you a frame slider (recommended when presenting — you control the pace), *Play animation* runs the GIF. If the `figures/frames/` folder is absent, the manual option is hidden and the GIF plays; that is designed fallback behaviour, not a fault.

---

### Slide 1 — System overview
**Figure:** `project_pipeline.png`

**What you see.** Nine boxes in three phases: prepare the data → design the inputs → learn and decide.

**How to read it.** Follow the arrows left to right, then down. Each box carries its own headline number, so the diagram doubles as a summary of scale: 1,296,675 transactions in, a ranked list of 908 customers out.

**Say this.** "The deliverable is an operational contact list, not a performance score. Every design decision downstream is judged by whether it improves that list."

**If asked — "is this the whole study?"** Be honest: it is the spine. It does not show the four validation experiments — the threshold sweep, the k-selection experiment, the ablation study, and the time-series baseline. Those appear on slides 7, 8, 11 and in the Model lab page.

---

### Slide 2 — Establishing a noise floor
**Figure:** `distributions.png`

**What you see.** Transaction amounts on a log scale, category shares, and the distribution of each customer's month-to-month spending variability.

**How to read it.** The third panel is the one that matters. The **median coefficient of variation is 0.38** — a typical customer's monthly spend swings ±38% around their own average, with nothing wrong.

**Say this.** "We measured normal variation before defining abnormal. A one-month rule would have flagged mostly noise, so the three-month window is a measured decision, not a convention."

**If asked — "why three months and not two or four?"** Two-month and four-month window shapes were tested (2+2 and 3+1) and rejected; the evidence is on the Label explorer page.

---

### Slide 3 — Explaining cohort growth
**Figure:** `spend_decomposition.png`

**What you see.** Total spend decomposed into active customers × transactions per customer × average ticket.

**How to read it.** Customer count is flat. Average ticket is flat. **Transactions per customer rose 42%** through 2019. The growth is uniform across the cohort, not driven by a subset.

**Say this.** "Because everyone was growing together, an absolute dollar threshold would have flagged almost nobody in early 2019 and almost everybody by 2020. That is why erosion is measured relative to the cohort median."

**If asked — "isn't the cohort-relative label just a normalisation trick?"** It is a deliberate design response to a measured property of the data. Slide 3 is the evidence that made it necessary.

---

### Slide 4 — Feasibility check
**Figure:** `persistence.png`

**What you see.** Two scatter plots: this month's log spend against next month's (r = 0.75), and one three-month average against the next (r = 0.86).

**How to read it.** Both correlations are strongly positive, so past spending does carry information about future spending. The rise from 0.75 to 0.86 happens because averaging three months cancels independent monthly noise.

**Say this.** "We ran this before building any features. Had the correlation been near zero, no model and no feature engineering could have recovered a signal that was not in the data."

**If asked — "if r = 0.75, why is the problem hard?"** Crucial distinction, and it appears again in §8d: 0.75 is **between-customer** level stability — big spenders stay big spenders. **Within** a customer, the median lag-1 autocorrelation is about −0.04, essentially zero. Predicting *level* is easy; predicting *change* is not.

---

### Slide 5 — Building training examples
**Figure:** `anim_sliding_window.gif` *(animated — use the frame slider)*

**What you see.** A three-month observation window and a three-month outcome window sliding across the calendar, twelve times.

**How to read it.** Each slide of the window produces one labelled example per customer: 908 × 12 = **10,896 rows**. Watch snapshots 7 and 8 drop out — their outcome windows overlap the test outcome windows in calendar time.

**Say this.** "The embargo costs us two of twelve snapshots, about 17% of the data. We accept that cost because it is what makes every number reported afterwards trustworthy."

**If asked — "isn't the same customer appearing twelve times a leak?"** No: the split is temporal, not random. A customer appears in both train and test, but at different times, and no outcome window crosses the boundary. Random splitting would have been the leak.

---

### Slide 6 — Validating the label
**Figure:** `label_quality.png`

**What you see.** Four label-quality criteria evaluated at each candidate threshold.

**How to read it.** At τ = 0.25: **97%** of labels are unchanged when using raw rather than winsorized spend (so the label is not an artefact of outlier capping); a flagged customer is **4.9×** more likely than average to be flagged again next period; prevalence is **7.2%**.

**Say this.** "Erosion is a persistent state, not one weak quarter. That persistence is what makes intervention meaningful — the condition lasts long enough to respond to."

**If asked — "why Jaccard overlap and not simple agreement?"** With 93% negatives, two label sets that disagree on every positive would still 'agree' about 93% of the time. Jaccard compares only the positive sets, so it cannot be inflated by the majority class.

---

### Slide 7 — Choosing the threshold
**Figure:** `anim_tau_sweep.gif` *(animated — use the frame slider)*

**What you see.** The threshold sliding from 0.10 to 0.45. Left panel: the gap distribution with the cutoff moving through it. Right panel: prevalence against persistence lift.

**How to read it.** As the threshold tightens, fewer customers are flagged but those flagged repeat more often — up to a point. At the extreme it reverses, because very large drops are one-off shocks rather than sustained decline. **0.25 sits where prevalence stays usable and persistence peaks.**

**Say this.** "The threshold was chosen on four criteria declared in advance, not by preference. We also report results at 0.20 and 0.30 so the reader can verify the conclusions hold either side of our choice."

**If asked — "why not optimise τ for model performance?"** That would be choosing the problem to suit the answer. The label must be defined by what makes business sense, then the model measured against it.

---

### Slide 8 — Customer segmentation
**Figure:** `segment_heatmap.png`

**What you see.** Seven segments as rows, behavioural features as columns, min-max scaled with the real values annotated.

**How to read it.** Read across a row to characterise a segment. seg_5 is high on spend and activity (affluent heavy spenders); seg_6 is low on everything (disengaged seniors). Clustering used **2019 data only**, so no future information entered the features.

**Say this.** "The number of segments was chosen by experiment — inner-validation PR-AUC across k = 2 to 7 — not by looking at an elbow plot. Seven added +0.055 PR-AUC."

**If asked — "the smallest segment has only 55 customers, is that acceptable?"** Yes, with a caveat you should volunteer: 55 customers is enough to be a distinct behavioural group, but any erosion rate computed on it carries wide uncertainty. That is exactly why seg_1's "zero erosion" is reported as *no evidence of erosion*, not *proven safe*.

**Important caution.** Cluster IDs are **assignment order, not identity** — k-prototypes numbers clusters by initialisation, so seg_3 in one run is not seg_3 in the next. The names in `SEGMENT_NAMES` are re-derived from the profile table after every run. Never quote a segment number without checking the profile table.

---

### Slide 9 — Why a tabular foundation model
**Figure:** `tabpfn_concept.png`

**What you see.** The size of the problem: **6,356 training rows, 548 positive cases, 55 features** — roughly ten events per variable, the floor of accepted practice.

**How to read it.** The figure leads with the constraint, not the architecture. A model trained from scratch must learn the entire problem from 548 positive examples.

**Say this.** "TabPFN is pre-trained on millions of synthetic tables and performs no gradient training on our data. Our table is supplied as context and prediction is a single forward pass. At this sample size, model family constrains performance more than feature count does."

**If asked — "is this just a bigger model?"** No. It is a different learning paradigm: in-context learning rather than fitting parameters to your data. The practical consequences are that it needs a GPU above 5,000 rows and cannot be inspected coefficient-by-coefficient — which is why the dashboard's explanation panels use a logistic model instead.

---

### Slide 10 — Model comparison
**Figure:** `zoo_curves.png`

**What you see.** ROC curves (left) and precision-recall curves (right) for all eight models.

**How to read it.** The ROC curves bunch together between 0.80 and 0.86 and tell you almost nothing — with 97.5% negatives, the true-negative term dominates and flatters everything. The PR curves separate properly, because precision ignores true negatives. TabPFN leads from roughly 0.1 to 0.45 recall, exactly where a realistic contact budget operates.

**Say this.** "TabPFN leads with PR-AUC 0.218 against a no-skill baseline of 0.025, ahead of logistic regression at 0.173. With about 69 positive test cases, differences below 0.03 are within sampling noise — so we present TabPFN as the leading model, not as conclusively the best."

**If asked — "at the 10% budget, three models tie. Which do you choose?"** They tie at *one point on the curve*. Across the whole ranking TabPFN leads by 0.045–0.058 PR-AUC, well outside the noise band, and it won all six ablation configurations. We chose it for robustness across budgets rather than for a single operating point.

---

### Slide 11 — Robustness of the result
**Figure:** `tabpfn_stability.png`

**What you see.** Three models × six feature configurations, PR-AUC. The chosen configuration is shaded; the two rejected ones are marked.

**How to read it.** TabPFN is highest in **all six** configurations and moves by only 0.015 across them — less than the noise band. XGBoost is erratic: it degrades as features widen (overfitting at 548 positives) yet peaks on the urbanicity set alone.

**Say this.** "The ranking is a property of the models, not of our particular feature choice. And because TabPFN is nearly indifferent to which features it receives, we do not select the feature set by its single highest cell — those differences are not distinguishable from noise."

**If asked — "why not use all 59 features?"** Adding urbanicity leaves TabPFN essentially unchanged (+0.001), costs logistic regression 0.008, and gains XGBoost 0.017 — all inside the noise band. Meanwhile 59 features drops us below ten events per variable. A feature that helps one model and harms two is noise, not signal, which matches the §5G-2 tests that found no association between city type and erosion.

---

### Slide 12 — Early warning for one customer
**Figure:** `anim_customer_story.gif` *(animated — use the frame slider)*

**What you see.** Customer 346208242862904's seventeen months revealed one at a time, with the model's risk score below.

**How to read it.** The score starts at **0.07** — indistinguishable from a healthy account — and reaches **1.00** by the final snapshot while the visible decline in the spend line is still small. That interval between the score rising and the decline becoming obvious is the actionable window.

**Say this.** "The value of the system is lead time. Operationally, how early a customer is identified matters as much as how often the identification is correct."

**If asked — "is 1.00 a 100% probability of erosion?"** No, and be quick to say so. These scores come from a class-balanced logistic model: they rank customers reliably but are **not calibrated probabilities**. 1.00 means "top of the ranking", not "certain".

---

### Slide 13 — Turning risk into a contact policy
**Figure:** `anim_budget_sweep.gif` *(animated — use the frame slider)*

**What you see.** The contact budget widening, with precision and recall responding.

**How to read it.**

| budget | contacts | eroders found | precision | lift |
|---|---|---|---|---|
| top 5% | 136 | 29 | 21.3% | 8.4× |
| top 10% | 272 | 41 | 15.1% | 6.0× |
| top 20% | 544 | 55 | 10.1% | 4.0× |

The interesting quantity is the **marginal** return: 5%→10% buys 12 extra eroders for 136 extra contacts; 10%→20% buys 14 for 272. Every step still beats random, but the return per contact halves.

**Say this.** "The model does not decide the budget. It converts a budget into an expected return, which leaves the commercial trade-off with the business rather than burying it in the analysis."

**If asked — "15% precision sounds poor."** Compare it to the 2.5% base rate: contacting 272 customers at random would find about 7 eroders; the model finds 41. That is six times better. Precision is low in absolute terms because the event itself is rare.

---

### Slide 14 — Allocating the budget
**Figure:** `segment_risk_overview.png`

**What you see.** Three panels: observed erosion rate per segment; model risk against actual erosion (bubble size = customers); total monthly value per segment with the at-risk share.

**How to read it.**
- **Panel 1** — seg_2 and seg_6 at **7.0%** each against a 2.5% base rate; seg_1 and seg_5 at zero.
- **Panel 2** — the model independently scores those same two highest (0.199 and 0.204, against 0.026–0.100 elsewhere). Unsupervised clustering and supervised learning agreeing is real confirmation. But note the untidy middle: **seg_3 erodes at 3.2% yet scores below seg_0, which erodes at 1.1%** — a genuine blind spot, worth volunteering.
- **Panel 3** — the money is **inverted**. The two hotspots are the smallest segments by value; the largest, seg_5 and seg_4, barely erode.

**Say this.** "Agreement between an unsupervised and a supervised method, arrived at independently, gives us enough confidence to act. But the segments that erode most carry the least revenue — we are losing customers, not money."

**If asked — "so is the system working?"** The honest answer: it predicts who will erode and names the segments they belong to, but those customers are the least valuable in the book. This is why value-weighted recall (55.7%) sits below headcount recall (59.4%). The natural next version weights the training objective by customer value.

---

## 3. The eleven interactive pages

Story mode is the argument. These pages are the evidence and the tools — use them to *answer* questions, not to present from. Each entry below gives the controls, what is actually computed, how to read the output, what to demonstrate, and the caution to state before you are asked.

---

### 3.1 Overview
*Orientation. Open here, spend thirty seconds, move on.*

**Controls:** none.

**What it shows.** Four headline metrics — behavioural customers (908), test-era erosion rate (2.5%), best model with its PR-AUC and its multiple of the no-skill baseline, and recall at a 10% contact budget. Below them, total cohort spend per month as a bar chart.

**How to read it.** The metric that matters is *"PR-AUC 0.218, 8.6× no-skill"*. A bare 0.218 sounds poor; against a 0.025 floor it is not. Always quote the ratio alongside the value.

**Point at:** the spend chart. Total spend roughly **doubles across 2019**. That ramp is the visual justification for the cohort-relative label — the single design decision the whole study rests on.

**Say:** "Everything in this application is read from cached results. The dashboard cannot produce a number that disagrees with the report."

---

### 3.2 Campaign simulator
*The main decision tool, and the best page to demo live.*

**Controls:** test snapshot (9 / 10 / 11, defaults to 11) · label definition (`erosion_20` / `erosion_25` / `erosion_30`, defaults to 25) · contact budget slider (1–30%, defaults to 10).

**What it computes.** Customers are sorted by risk; the top *k* = budget% are "contacted". Precision is the share of contacted customers who actually eroded; recall is the share of all eroders captured; **at-risk value covered** is the monetary version — the eroding customers' monthly value that falls inside the contacted set, divided by the total monthly value of all eroders.

**How to read the decile chart.** Note the axis: **decile 0 is the highest risk**, not the lowest (the code reverses `qcut`'s ordering). The dashed line is the base rate. Only the first two bars should stand clearly above it.

**Demo like this.** Set budget to 5%, then drag slowly to 20%. Precision falls from about 21% to 10% while recall climbs from 42% to 80%. Narrate the trade-off as it happens — it is far more convincing than the table in the report.

**Then scroll down.** The contact list is the actual deliverable: customer ID, risk, segment, monthly value. Download it as CSV in front of them. **Say:** "This is what the project produces — a list of people to call, not a metric."

**The τ selector is a hidden strength.** Switching the label to `erosion_20` or `erosion_30` re-scores everything against a different definition of erosion, live. If challenged on the 0.25 threshold, change it and show that the picture does not collapse.

**Caution to state first.** The `actually_eroded` column exists because we are evaluating on a held-out era where the truth is known. In production that column would be empty — it is shown here to demonstrate the ranking works, not because the system knows the future.

---

### 3.3 Cost–benefit
*The campaign in money. The most persuasive page for a business audience.*

**Controls:** snapshot · cost per contact ($1–500, default $10) · value saved per retained eroder ($10–20,000, default $500) · offer success rate (0.05–1.0, default 0.30).

**What it computes.** For every budget from 1% to 30%:

```
net benefit(k) = success rate × value saved × true positives caught − cost per contact × k
```

The optimum is the budget maximising that expression, marked with a red dot.

**How to read it.** The curve rises while marginal contacts are still precise and falls once the ranking runs out of true eroders. The zero line matters: budgets where the curve dips below it *lose money*.

**Demo like this.** Raise cost per contact to $50 — the optimum moves **left** (contact fewer, more precisely). Raise value saved to $5,000 — it moves **right** (worth casting a wider net). This shows the model doesn't dictate the policy; the economics do.

**Caution to state first.** Three of the four inputs are **assumptions, not measurements**. We never observed the true value of a retained customer or the success rate of an offer. The page shows how the decision *responds* to economics; it does not claim to know them.

---

### 3.4 Risk map
*Operational logistics, deliberately not an analytical finding.*

**Controls:** snapshot.

**What it shows.** One dot per customer on a US map. Colour runs green→red by **risk percentile** (not raw risk — percentile is used so the palette spreads even when scores cluster). Dot radius scales with monthly value. Hover for customer ID, risk, segment, value and state. Below the map, mean risk by state for the top 15.

**Caution — state this before showing the page, not after.** Geography has **no predictive value** in this simulated data. The §5G-2 tests found no association between city type and erosion (Cramér's V = 0.000, p = 0.75), and urbanicity was measured and rejected in the ablation. The state table *will* show variation between states; that variation is noise. This page exists for campaign logistics — which regional team calls whom — and nothing else.

**If the page errors** with "Location columns missing", `dashboard_data` was built by an old version of `prepare_dashboard_data.ipynb`. Regenerate it.

---

### 3.5 Alerts — risk movers
*Early warning in its purest form. The page a retention team would actually open on a Monday.*

**Controls:** snapshot.

**What it computes.** `risk_change = risk(t) − risk(t−1)` for every customer, sorted descending. Two metrics: how many customers have rising risk, and the largest single jump. Then the top 20 risers as a table, and their full risk trajectories across all snapshots as overlaid lines.

**How to read the trajectory chart.** You are looking for lines that climb steeply at the right-hand end. A customer sitting at constant high risk is already known; a customer *moving* is new information.

**Say:** "Everything else on this dashboard reports a state. This page reports a change — and a change is what you can act on before erosion completes."

---

### 3.6 Customer drill-down
*One customer under the microscope. Use this when asked "but why did the model flag that person?"*

**Controls:** snapshot · customer (the 50 highest-risk customers at that snapshot).

**What it shows.** Four metrics at the top: risk score, segment (with its name and a target/skip recommendation), monthly value, and whether the customer actually eroded. Then two charts side by side — the monthly spend trajectory with the **three observation months highlighted in blue** against grey history, and the risk history across snapshots on a fixed 0–1 axis.

**The "why flagged" panel** is the important one. For each feature it computes

```
contribution = coefficient × (value − mean) / standard deviation
```

and plots the twelve largest by absolute size. **Red bars push risk up, blue pull it down.** Because the model is linear, these contributions sum exactly to the score's linear term — this is a true decomposition, not an approximation like SHAP.

**Caution to state first.** This panel uses the **logistic model**, not TabPFN, because TabPFN has no inspectable coefficients. The explanations are directionally sound and come from the second-best model (PR-AUC 0.173 against 0.218), but they explain a *different* model from the headline result. That is a real limitation, and volunteering it is much stronger than being caught by it.

**Also note:** only the top 50 by risk are selectable, so you cannot browse a low-risk customer here for contrast.

---

### 3.7 What-if simulator
*Makes the model's logic tangible. Good for showing it responds to behaviour, not to noise.*

**Controls:** snapshot · customer · six sliders — total spend change (±50%), transaction count (±50%), distinct merchants (±50%), discretionary share (±20 percentage points), spend trend (±0.5), active days (±50%).

**What it computes.** It edits the chosen features and re-evaluates `sigmoid(intercept + Σ coefficient × standardised feature)` live. Spend is handled correctly in log space (`log1p(expm1(x) × (1 + pct))`), and discretionary share is clipped to [0, 1].

**Demo like this.** Cut spend −30%: risk rises. Add a positive spend trend: risk falls. Both are the direction a person would expect, which is the point — the model has learned something sensible, not an artefact.

**Caution to state first.** This is a **partial-derivative view, not a simulation**. Moving "transaction count" leaves the transaction *trend* features untouched, even though in reality the two move together. It answers "what does this coefficient do?", not "what would happen if this customer changed?" And, as with the drill-down, it is the logistic model.

---

### 3.8 Segment explorer
*The reference table. Use it to settle any "which segment is that?" question.*

**Controls:** none.

**What it shows.** Per segment: row count, observed erosion rate, mean model risk, the human-readable name, and a target/skip recommendation. A bar chart colours any segment above the base rate red. Below, a behavioural profile table — mean log spend, transactions per month, average ticket, discretionary share, night-time share and age for each segment in the test era.

**How to read it.** seg_2 and seg_6 at 7.0% are the hotspots, and the model's mean risk is highest for exactly those two (0.199 and 0.204, against 0.026–0.100 elsewhere). That agreement between an unsupervised method and a supervised one is the page's headline.

**Two cautions, both worth volunteering.**
1. The count column is **test rows, not customers** — 2,724 rows is 908 customers × 3 snapshots. Divide by three for the customer count (seg_1's 165 rows are 55 customers).
2. **Cluster IDs are assignment order, not identity.** k-prototypes numbers clusters by initialisation, so seg_3 in one run is not seg_3 in the next. The names are re-derived from the profile table after every run. Never quote a segment number without checking.

**And the honest wrinkle:** seg_3 erodes at 3.2% yet the model scores it 0.090 — below seg_0, which erodes at only 1.1%. The model under-rates seg_3. Say it before you are asked.

---

### 3.9 Model lab
*The evidence room. This is where you go when asked "how do you know your features do anything?"*

**Controls:** none — three stacked result sets.

**1. Eight-model benchmark.** Horizontal bars of PR-AUC with TabPFN highlighted, then the full table with ROC-AUC, PR-AUC, precision@10% and recall@10%. The caption states the no-skill baseline so nobody reads 0.218 without its reference point.

**2. Ablation study.** The full 18-row grid — three models × six feature configurations — including the `PR_gain_vs_baseline` column. This is where urbanicity's rejection is documented.

**3. Raw-lag baseline.** AR-3 uses the **same three months** as our engineered features with no engineering at all and collapses to 0.040–0.084 against TabPFN's 0.224. That single comparison is the answer to "does feature engineering earn its keep?" AR-6 rescues only the linear model, and on a smaller training set (3,632 rows against 6,356) — say so, because it is the one row that appears to contradict the conclusion.

---

### 3.10 Label explorer
*Where you go when asked "why 0.25?" or "why a 3+3 window?"*

**Controls:** a τ slider across the five candidate thresholds (0.15 / 0.20 / 0.25 / 0.30 / 0.40).

**What it shows.** Four metrics update as you slide — prevalence, winsorized-versus-raw agreement, Jaccard overlap of the positive sets, and persistence lift — followed by a written verdict for that threshold. Then the full criteria table, and a second table comparing alternative window shapes.

**Demo like this.** Slide to **0.15**: rejected, it sits at the noise floor of three-month averages. Slide to **0.40**: rejected, some snapshots have zero positives and repeat probability collapses — it captures one-off shocks, not disengagement. Slide back to **0.25**: prevalence in the predicted band, robust to winsorization, and flagged customers are about 5× likelier than baseline to be flagged again.

**Say:** "The label was engineered by experiment against four criteria declared in advance, not chosen by taste. And we report results at 0.20 and 0.30 so you can check the conclusions either side of our choice."

**The window-shape table** answers the other half: 2+2 and 3+1 inflate prevalence at every threshold, because less averaging means more noise crossings. 3+3 was retained on that evidence.

---

### 3.11 Drift monitor
*The production-thinking page. Shows you understand what happens after deployment.*

**Controls:** none.

**What it shows.** Erosion rate per snapshot, coloured by era — blue for train (snapshots 0–6), grey for the embargoed 7–8, red for test (9–11). Two metrics below give train-era prevalence (8.6%) against test-era prevalence (2.5%), with the drift as a delta.

**How to read it.** The drop is real and has a clear cause: the simulator's frequency ramp stabilises in 2020, so fewer customers cross a *relative* threshold. It is not a modelling error and not a bug in the label.

**Why it does not invalidate the results.** Ranking metrics remain valid under prevalence shift — but PR-AUC must always be compared to *its own era's* baseline, which is why every PR-AUC in this project is quoted against 0.025 rather than against the training prevalence.

**Say:** "In production this page would drive quarterly threshold recalibration. Drift is not something to hide; it is something to monitor."

---

## 4. Suggested presentation routes

**Ten minutes (full defence).** Story slides 1 → 2 → 3 → 5 → 7 → 9 → 10 → 11 → 13 → 14, then live-demo the Campaign simulator. This covers the design decisions, the evidence, and the deliverable.

**Three minutes (if time is cut).** Slide 1 (what it is) → slide 10 (it works) → slide 13 (what it buys) → Campaign simulator (here is the list).

**If challenged on rigour:** Model lab and Label explorer. Both exist specifically to show that decisions were tested rather than asserted.

---

## 5. Limitations to volunteer, not defend

Stating these first is far stronger than conceding them under questioning.

1. **The data is simulated.** Sparkov generates plausible transactions but models no household budget constraint — which is why the Engel-curve validation of the LLM features failed, and why the median inter-purchase interval of 4.6 hours is unrealistic.
2. **Only 69 positive test cases.** Every PR-AUC difference below roughly 0.03 is sampling noise. We never claim a winner inside that band.
3. **Scores rank, they do not calibrate.** A 1.00 means "top of the ranking", not "certain to erode".
4. **Explanations come from the logistic model**, not from TabPFN.
5. **The system saves customers, not revenue.** Erosion concentrates among low-value segments; value-weighted recall (55.7%) is below headcount recall (59.4%).
6. **No churn exists in this data.** All 908 customers are active in all 17 months, so the problem is gradual relative erosion, not account closure. This is stated in the report rather than glossed.

---

## 6. Files the dashboard needs

**`figures/` — 14 files, next to `app.py`:**
`project_pipeline.png` · `distributions.png` · `spend_decomposition.png` · `persistence.png` · `anim_sliding_window.gif` · `label_quality.png` · `anim_tau_sweep.gif` · `segment_heatmap.png` · `tabpfn_concept.png` · `zoo_curves.png` · `tabpfn_stability.png` · `anim_customer_story.gif` · `anim_budget_sweep.gif` · `segment_risk_overview.png`

Optional: `figures/frames/<gif name>/` containing the split PNG frames, which enables the manual frame slider. Without it, the app falls back to GIF playback.

**`dashboard_data/` — 5 files:**
`risk_scores.parquet` · `monthly_spend.parquet` · `model_data.parquet` · `results.json` · `explain_model.json`

Rebuild these with `prepare_dashboard_data.ipynb` whenever the notebook is re-run.

> **Note:** `project_pipeline.png` was originally hand-made and is not produced by any notebook cell, so a "regenerate all figures" sweep will silently drop it. A generating cell now exists in the presentation-graphics section — use it, so the file comes out of the pipeline like everything else.
