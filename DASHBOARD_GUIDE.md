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

Story mode is the argument. These pages are the evidence and the tools — use them to answer questions, not to present from.

### Overview
**For:** orientation. Cohort size, test-era erosion rate, best-model performance, cohort monthly spend.
**Point at:** the 2019 spending ramp in the spend chart. It is the visual justification for the cohort-relative label.

### Campaign simulator
**For:** the main decision tool, and the best page to demo live.
**Do this:** pick a snapshot, then drag the contact-budget slider and let them watch precision fall while recall rises. The table below is the actual contact list, downloadable as CSV.
**Say:** "This is the deliverable — an actual list of customers to call, not a metric."

### Cost-benefit
**For:** the campaign in money terms. Enter cost per contact, value saved per retained eroder, and offer success rate.
**Do this:** change the economics and watch the optimum budget move. It is the most persuasive page for a business audience.
**Caution:** the inputs are *your* assumptions, not measured quantities. Say so before someone asks.

### Risk map
**For:** operational planning — where the at-risk customers physically are.
**Caution, and state it unprompted:** geography has **no** predictive value in this simulated data. §5G-2 found no association between city type and erosion, and urbanicity was rejected in the ablation. This page is a logistics view, not a finding.

### Alerts — risk movers
**For:** early warning in its purest form: customers whose risk rose most since the previous snapshot.
**Say:** "A rising score is actionable before the erosion completes. This is the page a retention team would actually open on a Monday morning."

### Customer drill-down
**For:** one customer under the microscope — spend trajectory with the observation window highlighted, risk history, and a "why flagged" panel giving each feature's signed contribution.
**Caution:** the explanation panel uses the **logistic** model, not TabPFN, because TabPFN has no inspectable coefficients. Say this before you are caught: the explanations are directionally informative but come from a different (and slightly weaker) model than the headline result.

### What-if simulator
**For:** demonstrating that the model responds sensibly to behaviour, not to noise.
**Do this:** cut spend by 30% and watch risk rise; add a positive spend trend and watch it fall.
**Caution:** recomputed live from logistic coefficients — again, not TabPFN.

### Segment explorer
**For:** the segment profile table and per-segment erosion, in a form you can sort and inspect.
**Use it to** answer any "which segment is that?" question — and to check the cluster-ID caution from slide 8.

### Model lab
**For:** the evidence room. The eight-model benchmark, the ablation grid, and the raw-lag time-series baseline.
**This is where you go** when asked "how do you know your features are doing anything?" The AR-3 baseline uses the *same three months* as our features with no engineering and collapses to 0.040–0.084 against 0.224. That comparison is the answer.

### Label explorer
**For:** how the erosion definition was chosen — five candidate thresholds against four criteria, plus alternative window shapes.
**This is where you go** when asked "why 0.25?" or "why 3+3?".

### Drift monitor
**For:** erosion rate per snapshot, split into train / embargoed / test eras.
**Explains** the prevalence drift from 8.6% in training to 2.5% in test: the simulator's growth ramp calms down in 2020, so fewer customers cross a relative threshold.
**Say:** "In production this page would drive quarterly threshold recalibration. Drift is not a bug to hide, it is a thing to monitor."

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
