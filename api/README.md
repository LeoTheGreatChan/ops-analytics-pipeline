# API layer (in progress)

This folder will hold the Flask API that serves `/data/processed/*.json` as live endpoints, replacing the inline data currently embedded in `frontend/index.html`.

Planned endpoints:

- `GET /api/kpis`
- `GET /api/segments`
- `GET /api/insights?objective=customer_satisfaction|cost_reduction`
- `GET /api/forecast`

Deployment target: Render (free tier). See main README for the full architecture rationale.
