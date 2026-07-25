# Predictive Operations Analytics Pipeline
### Statistical Insight Engine · Delay-Risk Model · Objective-Ranked Recommendations

**Live Demo →** [leothegreatchan.github.io/ops-analytics-pipeline/](https://leothegreatchan.github.io/ops-analytics-pipeline/)

---

## The Problem

Most operational dashboards stop at description: here is what happened. They rarely tell an operations team which finding matters most, what to do about it, or how the recommendation changes depending on whether the business is optimising for customer experience or cost control this quarter.

This project builds the layer above the dashboard: a pipeline that turns raw delivery data into ranked, numeric, business-objective-aware recommendations, with the reasoning behind each one traceable back to a real statistic.

---

## Data

**Source:** [Amazon Delivery Dataset](https://www.kaggle.com/datasets/sujalsuthar/amazon-delivery-dataset) (Kaggle), a real-world last-mile delivery dataset, 43,739 rows covering Feb–Apr 2022.

**This is a static, historical dataset, not a live feed.** Every claim in this project is framed accordingly: "built on a real-world delivery dataset," not "live monitoring." The distinction matters and is kept precise throughout the dashboard, the code comments, and this README.

**Columns:** Order_ID, Agent_Age, Agent_Rating, Store/Drop Latitude & Longitude, Order_Date, Order_Time, Pickup_Time, Weather, Traffic, Vehicle, Area, Delivery_Time, Category.

---

## Data Quality & Validation

This section exists because catching problems in the data is as much a part of the analysis as the findings themselves, and it's worth showing the working, not just the conclusions.

**Cleaning:** 43,739 raw rows → 43,594 after dropping 145 rows (0.3%) with nulls in Weather, Traffic, or Agent_Rating. 3,485 rows (8.0%) had unusable geocodes (near 0,0, clearly invalid for an India-based dataset) and were excluded from distance-dependent calculations specifically, while still contributing to every other statistic.

**Confounded variable caught and removed:** an early pass flagged "Morning deliveries" as showing a dramatic 93% lower breach rate. Investigation showed this wasn't an independent finding: 100% of Morning-band orders in this dataset also carry `Traffic = Low`, a structural artifact of the source data, not a discovered pattern. Reporting both would have double-counted the same underlying signal. Time_Band was excluded as a standalone insight dimension as a result, kept for descriptive display only.

**An assumption corrected against the real data:** external analysis of this dataset is sometimes cited as showing no meaningful relationship between agent rating and delivery performance. The actual correlation in this dataset is -0.31, and the breach rate drops from 76.9% for agents rated below 3.0 to roughly 14-16% for agents rated 4.5 and above, a real, verified, and substantial effect once checked directly rather than taken on secondary authority. It became the model's top-ranked feature.

**Holiday sample size:** only two public holidays fall within the dataset's 8-week window (Maha Shivaratri, Holi), landing on only two of seven weekdays. The holiday effect (23.7% vs 22.9% breach rate) is reported as directional, not statistically robust, rather than dressed up as a confident finding the sample can't support.

---

## Pipeline Architecture

**① Feature engineering** — `Distance_km` (haversine from store/drop coordinates, nulled for bad geocodes), `Weekday`, `Is_Weekend`, `Is_Holiday` (India public holiday calendar), `Order_Hour` bucketed into time-of-day bands, `SLA_Breach` flag (relative threshold: 75th percentile of delivery time, computed per Area rather than a fixed arbitrary cutoff).

**② Statistical insight engine** — scans single and compound segments (Area, Weather, Traffic, Vehicle, Weekday, Area×Traffic, Weather×Traffic, Agent Rating bands), filters to segments with a minimum sample size (200 rows) and a meaningful effect size (±15% breach-rate lift vs baseline), and computes a numeric buffer suggestion per segment from the actual 90th-percentile-vs-median delivery time gap, not an invented number.

**③ Predictive model** — Random Forest classifier predicting delay risk per delivery. 85.9% accuracy, 0.74 F1 on the breach class (87% recall, 64% precision — deliberately tuned via `class_weight='balanced'` to catch real breaches over minimising false alarms, since missing a genuine SLA breach is operationally costlier than over-flagging one). Feature importances cross-validated against the statistical layer for consistency (both independently found the holiday effect negligible, for example).

**④ Business-value ranking** — every candidate finding is scored against two weighted objectives, Customer Satisfaction (severity-weighted) and Cost Reduction (volume-weighted), producing genuinely different top-ranked recommendations, not a cosmetic reorder. For example: Customer Satisfaction ranks a severe but lower-volume weather×traffic combination first; Cost Reduction ranks a modest-severity but very high-volume vehicle category first, because its cumulative cost exposure is larger.

**⑤ Insight write-up** — the ranked statistical findings are converted into plain-English, numbered recommendations with bolded figures (e.g. "Fog weather combined with Jam traffic shows a **174% higher** breach rate than average... Add a **31% buffer**"), an LLM writing layer on top of a stats/ML layer, kept explicitly distinct rather than blurred into a single "AI insight" claim.

**⑥ Dashboard** — four-tab interface (Overview, Segments, Insights & Recommendations, Risk Forecast), with an objective toggle that live-reorders the recommendation list, and a colour system (green = beneficial, black = neutral, red = risk) applied consistently across KPI values, charts, and table cells.

---

## What's ML, What's Statistics, and What's an LLM Writing It Up

Being precise about this distinction is itself part of the deliverable:

| Component | Technique | Type |
|---|---|---|
| Delay-risk classifier | Random Forest | Supervised ML |
| Numeric buffer suggestions | Percentile/quantile statistics | Statistics, not ML |
| Segment findings | Grouped comparison + effect-size filtering | Statistics, not ML |
| Objective-based ranking | Weighted scoring function | Business logic, not ML |
| Plain-English write-up | LLM call | AI, explicitly a writing layer, not a decision-maker |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data processing | Python, Pandas, NumPy |
| Modelling | scikit-learn (Random Forest) |
| Statistics | SciPy, Pandas |
| Insight generation | LLM API call (batch, precomputed — not per-request) |
| API (Phase 2) | Flask, Flask-CORS |
| Frontend | HTML, CSS, Chart.js |
| Hosting (planned) | GitHub Pages (frontend) + Render (API) |

---

## Key Design Decisions

**Why precompute instead of computing live per request?** The underlying dataset never changes. Recomputing statistics, retraining the model, or calling an LLM on every page visit would be slower, costlier, and would misrepresent the project as live analysis when it isn't. Everything is computed once in a batch step and served as static JSON; the only runtime work is filtering and re-sorting already-computed data, which is why the objective toggle feels instant.

**Why a relative SLA threshold instead of a fixed one?** The dataset has no pre-labelled "late" flag. A fixed threshold (e.g. "over 150 minutes is late") would be an arbitrary number invented for this project. A relative threshold, the 75th percentile of delivery time within each area, is data-derived and defensible under questioning.

**Why keep n8n out of the core pipeline?** n8n's role here is orchestration (scheduled refresh, alert routing), not analysis. The actual value of this project, statistical rigour, model validation, business-aware ranking, lives entirely in the Python layer. Leading with n8n would blur this project's positioning with the author's separate [sentiment analysis pipeline](https://github.com/LeoTheGreatChan/saas-sentiment-analyzer), which is genuinely n8n-centric. n8n is planned for Phase 2 as a scheduled refresh and alert layer only.

---

## Reproducing This Pipeline

1. Download the dataset (see `data/raw/README.md`) and place `amazon_delivery.csv` in `data/raw/`
2. `pip install -r requirements.txt`
3. Run scripts in order from the repo root:
   ```
   python scripts/feature_engineering.py
   python scripts/insight_engine.py
   python scripts/model.py
   python scripts/build_insights.py
   python scripts/build_remaining_data.py
   ```
4. Outputs land in `data/processed/` as five JSON files, currently embedded directly into `frontend/index.html`; Phase 2 will serve these live via the Flask API instead

---

## Extension Opportunities

- **Live API layer** (Phase 2, in progress) — Flask endpoints replacing the inline data currently embedded in the frontend
- **n8n refresh/alert layer** (Phase 2) — scheduled reruns and Slack/email alerts for newly high-risk segments
- **Second dataset partner** — extending the pipeline to a multi-carrier delivery dataset to test whether findings generalise
- **Time-series forecasting** — a proper Prophet or seasonal-decomposition model for the Risk Forecast tab, beyond the current pattern-based aggregation

---

*Built by Leo Chan · July 2026*
