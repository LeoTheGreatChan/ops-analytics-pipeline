"""
Statistical ranking + insight write-up.

Refactored in Phase 2 into importable functions so both this standalone
script (for local/manual runs) and api/app.py (for the /api/refresh
endpoint) share one source of truth for ranking logic, no duplicated code
between the two.

Write-up behaviour: if ANTHROPIC_API_KEY is set, uses the real Claude API
(scripts/llm_insight_writer.py). If not set, falls back to the deterministic
template function below, with a printed warning, this keeps local
development/testing free and working without a key, while production
(Render, with the key set as an environment variable) always uses the real
LLM call.
"""
import pandas as pd
import numpy as np
import json
import pickle
import os
import sys

sys.path.append(os.path.dirname(__file__))


def build_all_findings():
    """Loads Phase 1 outputs, adds the Agent_Rating dimension (numeric,
    bucketed, not part of the original categorical segment scan), and
    computes both objective scores. Returns (all_findings_df, overall_breach)."""
    df = pd.read_pickle('../data/processed/cleaned_data.pkl')
    findings = pd.read_pickle('../data/processed/findings.pkl')

    overall_breach = df['SLA_Breach'].mean()
    overall_delay = df['Delivery_Time'].mean()

    df['Agent_Rating_Bucket'] = pd.cut(df['Agent_Rating'], bins=[0, 3, 4, 4.5, 4.8, 5.1],
                                        labels=['Below 3.0', '3.0-4.0', '4.0-4.5', '4.5-4.8', '4.8-5.0'])
    rating_grp = df.groupby('Agent_Rating_Bucket', observed=True).agg(
        n=('SLA_Breach', 'size'), breach_rate=('SLA_Breach', 'mean'),
        avg_delay=('Delivery_Time', 'mean'), p90_delay=('Delivery_Time', lambda x: x.quantile(0.9)),
        median_delay=('Delivery_Time', 'median')
    ).reset_index()
    rating_grp = rating_grp[rating_grp['n'] >= 200].copy()
    rating_grp['dimension'] = 'Agent_Rating'
    rating_grp['segment_label'] = rating_grp['Agent_Rating_Bucket'].astype(str)
    rating_grp['breach_lift_pct'] = (rating_grp['breach_rate'] - overall_breach) / overall_breach * 100
    rating_grp['delay_diff_min'] = rating_grp['avg_delay'] - overall_delay
    rating_grp['suggested_buffer_pct'] = ((rating_grp['p90_delay'] - rating_grp['median_delay']) / rating_grp['median_delay'] * 100).round(0)
    rating_grp['suggested_buffer_min'] = (rating_grp['p90_delay'] - rating_grp['median_delay']).round(0)
    rating_grp = rating_grp[['dimension', 'segment_label', 'n', 'breach_rate', 'breach_lift_pct',
                              'avg_delay', 'delay_diff_min', 'suggested_buffer_pct', 'suggested_buffer_min']]
    rating_findings = rating_grp[rating_grp['breach_lift_pct'].abs() >= 15].copy()

    all_findings = pd.concat([findings, rating_findings], ignore_index=True)
    n_norm = (all_findings['n'] - all_findings['n'].min()) / (all_findings['n'].max() - all_findings['n'].min())
    lift_norm = (all_findings['breach_lift_pct'].abs() - all_findings['breach_lift_pct'].abs().min()) / \
                (all_findings['breach_lift_pct'].abs().max() - all_findings['breach_lift_pct'].abs().min())
    all_findings['score_customer_satisfaction'] = (0.75 * lift_norm + 0.25 * n_norm).round(3)
    all_findings['score_cost_reduction'] = (0.35 * lift_norm + 0.65 * n_norm).round(3)

    return all_findings, overall_breach


def rank_top_segments(all_findings, objective, top_n=6):
    """Returns the top-N ranked dataframe for the given objective, plus the
    #1 segment's identifier string (used by change_detection.py to check if
    the leader changed run to run)."""
    obj_col = f'score_{objective}'
    ranked = all_findings.sort_values(obj_col, ascending=False).head(top_n).reset_index(drop=True)
    top_segment_id = f"{ranked.iloc[0]['dimension']}::{ranked.iloc[0]['segment_label']}" if len(ranked) else None
    return ranked, top_segment_id


def _write_insight_template(row, overall_breach):
    """Fallback deterministic template, used only when no ANTHROPIC_API_KEY
    is set (local dev without incurring API cost). Not used in production."""
    seg = row['segment_label']
    dim = row['dimension']
    lift = row['breach_lift_pct']
    rate = row['breach_rate'] * 100
    buf = row['suggested_buffer_pct']
    n = int(row['n'])
    direction = "higher" if lift > 0 else "lower"

    if dim == 'Weather x Traffic':
        w, t = seg.split(' + ')
        finding = f"{w} weather combined with {t} traffic shows a **{abs(lift):.0f}% {direction}** breach rate than average ({rate:.1f}% vs {overall_breach*100:.1f}% baseline)."
        action = f"Add a **{buf:.0f}% buffer** to delivery estimates for this combination." if lift > 0 else "No buffer adjustment needed; this combination consistently outperforms baseline."
    elif dim == 'Area x Traffic':
        a, t = seg.split(' + ')
        finding = f"{a} deliveries during {t} traffic show a **{abs(lift):.0f}% {direction}** breach rate than average ({rate:.1f}% vs {overall_breach*100:.1f}% baseline)."
        action = f"Add a **{buf:.0f}% buffer** for routes in this segment." if lift > 0 else "Standard SLA buffer is sufficient for this segment."
    elif dim == 'Agent_Rating':
        finding = f"Agents rated **{seg}** show a **{abs(lift):.0f}% {direction}** breach rate than average ({rate:.1f}% vs {overall_breach*100:.1f}% baseline)."
        action = "Prioritise training or route reassignment for lower-rated agents." if lift > 0 else "No action needed; this rating band performs at or above target."
    else:
        finding = f"{dim} = {seg} shows a **{abs(lift):.0f}% {direction}** breach rate than average ({rate:.1f}% vs {overall_breach*100:.1f}% baseline)."
        action = f"Add a **{buf:.0f}% buffer** for this segment." if lift > 0 else "This segment consistently outperforms baseline; no action needed."

    impact_pp = abs(rate - overall_breach * 100)
    return finding + " " + action, f"estimated impact: {'-' if lift>0 else '+'}{impact_pp:.0f}pp breach rate · {n:,} deliveries/period affected"


def write_insights_json(ranked_df, objective, overall_breach, output_dir='../data/processed'):
    """Generates the write-up (real LLM if ANTHROPIC_API_KEY is set, template
    fallback otherwise) and writes insights_{objective}.json."""
    use_llm = bool(os.environ.get('ANTHROPIC_API_KEY'))

    if use_llm:
        from llm_insight_writer import write_insight_llm
    else:
        print(f"WARNING: ANTHROPIC_API_KEY not set — using template fallback for {objective}, "
              f"not the real LLM call. Set the key in production (Render env var).")

    out = []
    for i, row in ranked_df.iterrows():
        if use_llm:
            finding_text = write_insight_llm(row, overall_breach)
            rate = row['breach_rate'] * 100
            impact_pp = abs(rate - overall_breach * 100)
            impact_text = f"estimated impact: {'-' if row['breach_lift_pct']>0 else '+'}{impact_pp:.0f}pp breach rate · {int(row['n']):,} deliveries/period affected"
        else:
            finding_text, impact_text = _write_insight_template(row, overall_breach)

        out.append({
            "rank": i + 1,
            "top_priority": i == 0,
            "dimension": row['dimension'],
            "segment": row['segment_label'],
            "finding": finding_text,
            "impact": impact_text,
            "n": int(row['n']),
            "breach_rate": round(row['breach_rate'] * 100, 1),
            "breach_lift_pct": round(row['breach_lift_pct'], 1)
        })

    with open(f'{output_dir}/insights_{objective}.json', 'w') as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == '__main__':
    all_findings, overall_breach = build_all_findings()
    for objective in ['customer_satisfaction', 'cost_reduction']:
        ranked, top_segment_id = rank_top_segments(all_findings, objective)
        out = write_insights_json(ranked, objective, overall_breach)
        print(f"=== {objective.upper()} — top 3 preview ===")
        for item in out[:3]:
            print(f"#{item['rank']} {'[TOP PRIORITY] ' if item['top_priority'] else ''}{item['finding']}")
            print(f"   {item['impact']}")
        print()
    print("Full insight JSON files written for both objectives.")
