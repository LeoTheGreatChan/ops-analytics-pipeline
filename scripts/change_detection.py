"""
Change detection — Phase 2

Decides whether a refresh's new findings are different enough from the prior
run to justify (a) paying for a fresh LLM write-up and (b) firing an n8n
alert. Most refreshes on a stable operation should NOT cross this bar, most
weeks should be a quiet, free, silent data refresh. This is what keeps the
"conditional LLM call" honest rather than just calling it every time anyway.

Rule: material change = the max breach-rate-lift shift across all reported
segments exceeds MATERIAL_SHIFT_THRESHOLD_PP, OR the #1-ranked segment
(under either objective) is no longer the same segment as last run.
"""

import json
import os

MATERIAL_SHIFT_THRESHOLD_PP = 5.0  # percentage points
SNAPSHOT_PATH = "../data/processed/findings_prior_snapshot.json"


def _load_prior_snapshot():
    if not os.path.exists(SNAPSHOT_PATH):
        # first run ever — nothing to compare against, treat as material
        # so the first real refresh always generates fresh write-ups
        return None
    with open(SNAPSHOT_PATH) as f:
        return json.load(f)


def _save_snapshot(findings_df, top_cs_segment, top_cr_segment):
    snapshot = {
        "segments": {
            f"{row.dimension}::{row.segment_label}": round(row.breach_lift_pct, 1)
            for row in findings_df.itertuples()
        },
        "top_customer_satisfaction": top_cs_segment,
        "top_cost_reduction": top_cr_segment,
    }
    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(snapshot, f, indent=2)


def check_material_change(findings_df, top_cs_segment, top_cr_segment):
    """
    findings_df: the current run's findings (from insight_engine.py), must
                 have columns dimension, segment_label, breach_lift_pct
    top_cs_segment / top_cr_segment: string identifiers of this run's #1
                 ranked segment under each objective, for the "did the
                 leader change" check
    Returns: dict with material_change (bool), reason (str), max_shift_pp (float)
    """
    prior = _load_prior_snapshot()

    if prior is None:
        _save_snapshot(findings_df, top_cs_segment, top_cr_segment)
        return {
            "material_change": True,
            "reason": "First run — no prior snapshot to compare against.",
            "max_shift_pp": None,
        }

    current = {
        f"{row.dimension}::{row.segment_label}": row.breach_lift_pct
        for row in findings_df.itertuples()
    }

    shifts = []
    for key, current_lift in current.items():
        prior_lift = prior["segments"].get(key)
        if prior_lift is not None:
            shifts.append(abs(current_lift - prior_lift))
    max_shift = max(shifts) if shifts else 0.0

    leader_changed = (
        top_cs_segment != prior.get("top_customer_satisfaction")
        or top_cr_segment != prior.get("top_cost_reduction")
    )

    material = (max_shift >= MATERIAL_SHIFT_THRESHOLD_PP) or leader_changed

    if material:
        reasons = []
        if max_shift >= MATERIAL_SHIFT_THRESHOLD_PP:
            reasons.append(f"max segment shift {max_shift:.1f}pp (threshold {MATERIAL_SHIFT_THRESHOLD_PP}pp)")
        if leader_changed:
            reasons.append("top-ranked segment changed under at least one objective")
        reason = "; ".join(reasons)
    else:
        reason = f"Max shift {max_shift:.1f}pp, below {MATERIAL_SHIFT_THRESHOLD_PP}pp threshold; no leadership change."

    _save_snapshot(findings_df, top_cs_segment, top_cr_segment)

    return {"material_change": material, "reason": reason, "max_shift_pp": round(max_shift, 1)}
