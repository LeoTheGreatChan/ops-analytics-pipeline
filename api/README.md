# API layer

Flask API serving the pipeline's data and handling automated refreshes.

**Endpoints:**

- `GET /api/kpis`
- `GET /api/segments`
- `GET /api/insights?objective=customer_satisfaction|cost_reduction`
- `GET /api/forecast`
- `POST /api/refresh` — accepts a new CSV (multipart form field `file`), re-runs the pipeline, and returns drift-check and change-detection results. See main README's "Phase 2: Automated Refresh Pipeline" section for the full orchestration logic.

**Environment variables required in deployment:**

- `ANTHROPIC_API_KEY` — for the real LLM insight write-up call. Without it, `/api/refresh` falls back to a deterministic template (with a printed warning), useful for local development, not intended for production.

**Run locally:**

```
cd api
pip install -r ../requirements.txt
python app.py
```

**Deployment target:** Render (free tier). Set `ANTHROPIC_API_KEY` as a secret environment variable in Render's dashboard, never commit it to the repo.
