"""
Flask API — Phase 2

Serves the five precomputed JSON files (kpis, segments, insights x2,
forecast) as live endpoints, and exposes /api/refresh, which accepts a new
CSV of the same structure and re-runs the pipeline against it.

Orchestration in /api/refresh, in order:
  1. Save uploaded CSV, run feature_engineering.py (subprocess) to clean it
  2. Validate the EXISTING fixed model against the new batch (drift_check.py)
     — does not retrain, only flags if retraining should be considered
  3. Run insight_engine.py (subprocess) to rescan segments on the new data
  4. Rank findings per objective (build_insights.py functions)
  5. Check whether the new findings are materially different from the prior
     run (change_detection.py) — this gates steps 6 and 7
  6. If material: generate fresh LLM write-ups (real Claude API call) and
     signal that an alert should fire
  7. If not material: keep the previous run's insight write-ups as-is,
     skip the LLM call entirely (this is what keeps the LLM cost genuinely
     conditional rather than firing on every refresh regardless)
  8. Always regenerate kpis/segments/forecast (cheap, no LLM involved)

Run locally: `python app.py` (from the /api directory)
Deploy: Render, with ANTHROPIC_API_KEY set as an environment variable secret
"""

import os
import sys
import subprocess
import json

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from drift_check import check_drift
from change_detection import check_material_change
from build_insights import build_all_findings, rank_top_segments, write_insights_json
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'scripts')


def _load_json(name):
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


# ---- READ ENDPOINTS ----

@app.route('/api/kpis')
def kpis():
    return jsonify(_load_json('kpis.json'))


@app.route('/api/segments')
def segments():
    return jsonify(_load_json('segments.json'))


@app.route('/api/insights')
def insights():
    objective = request.args.get('objective', 'customer_satisfaction')
    if objective not in ('customer_satisfaction', 'cost_reduction'):
        return jsonify({'error': "objective must be 'customer_satisfaction' or 'cost_reduction'"}), 400
    return jsonify(_load_json(f'insights_{objective}.json'))


@app.route('/api/forecast')
def forecast():
    return jsonify(_load_json('forecast.json'))


# ---- REFRESH ENDPOINT (n8n calls this) ----

@app.route('/api/refresh', methods=['POST'])
def refresh():
    if 'file' not in request.files:
        return jsonify({'error': 'no file uploaded (expected multipart form field "file")'}), 400

    file = request.files['file']
    save_path = os.path.join(RAW_DIR, 'amazon_delivery.csv')
    file.save(save_path)

    # Step 1: feature engineering
    result = subprocess.run(
        ['python3', os.path.join(SCRIPTS_DIR, 'feature_engineering.py')],
        capture_output=True, text=True, cwd=SCRIPTS_DIR
    )
    if result.returncode != 0:
        return jsonify({'error': 'feature engineering failed', 'detail': result.stderr}), 500

    cleaned_df = pd.read_pickle(os.path.join(DATA_DIR, 'cleaned_data.pkl'))

    # Step 2: validate the fixed model against the new batch (no retraining)
    drift_result = check_drift(cleaned_df, model_path=os.path.join(DATA_DIR, 'model.pkl'))

    # Step 3: statistical insight scan on the new data
    result = subprocess.run(
        ['python3', os.path.join(SCRIPTS_DIR, 'insight_engine.py')],
        capture_output=True, text=True, cwd=SCRIPTS_DIR
    )
    if result.returncode != 0:
        return jsonify({'error': 'insight engine failed', 'detail': result.stderr}), 500

    # Step 4: rank per objective
    all_findings, overall_breach = build_all_findings()
    ranked_cs, top_cs_segment = rank_top_segments(all_findings, 'customer_satisfaction')
    ranked_cr, top_cr_segment = rank_top_segments(all_findings, 'cost_reduction')

    # Step 5: change detection gate
    change_result = check_material_change(all_findings, top_cs_segment, top_cr_segment)

    # Step 6/7: conditional write-up
    if change_result['material_change']:
        write_insights_json(ranked_cs, 'customer_satisfaction', overall_breach)
        write_insights_json(ranked_cr, 'cost_reduction', overall_breach)
        writeup_action = 'regenerated (material change detected)'
    else:
        writeup_action = 'skipped (no material change — prior write-up retained)'

    # Step 8: always refresh KPIs/segments/forecast (cheap, deterministic)
    result = subprocess.run(
        ['python3', os.path.join(SCRIPTS_DIR, 'build_remaining_data.py')],
        capture_output=True, text=True, cwd=SCRIPTS_DIR
    )
    if result.returncode != 0:
        return jsonify({'error': 'build_remaining_data failed', 'detail': result.stderr}), 500

    return jsonify({
        'status': 'refreshed',
        'rows_processed': len(cleaned_df),
        'drift_check': drift_result,
        'change_detection': change_result,
        'insight_writeup': writeup_action,
        'alert_recommended': change_result['material_change'] or drift_result['drift_detected']
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
