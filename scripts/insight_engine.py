"""
Statistical insight engine.

Refactored (Render-deployment fix) into a callable function so app.py can
call it in-process instead of via subprocess, same reasoning as
feature_engineering.py: avoids a second pandas/numpy load competing for
memory on Render's free tier.
"""
import pandas as pd
import numpy as np

MIN_SAMPLE = 200  # segments below this are too thin to report confidently

SEGMENT_DEFS = {
    'Area': ['Area'],
    'Weather': ['Weather'],
    'Traffic': ['Traffic'],
    'Vehicle': ['Vehicle'],
    # Time_Band excluded as a standalone finding: fully confounded with Traffic
    # (100% of 'Morning' band rows are Traffic='Low' in this dataset), so it
    # would double-report the same signal as the Traffic finding. Kept in the
    # dataframe for descriptive display only, not ranked as an independent insight.
    'Weekday': ['Weekday'],
    'Area x Traffic': ['Area', 'Traffic'],
    'Weather x Traffic': ['Weather', 'Traffic'],
    'Area x Weather': ['Area', 'Weather'],
}


def run_insight_engine(cleaned_data_path='../data/processed/cleaned_data.pkl',
                        output_path='../data/processed/findings.pkl',
                        verbose=True):
    df = pd.read_pickle(cleaned_data_path)

    overall_breach = df['SLA_Breach'].mean()
    overall_delay = df['Delivery_Time'].mean()

    candidates = []
    for dim_name, cols in SEGMENT_DEFS.items():
        grouped = df.groupby(cols).agg(
            n=('SLA_Breach', 'size'),
            breach_rate=('SLA_Breach', 'mean'),
            avg_delay=('Delivery_Time', 'mean'),
            p90_delay=('Delivery_Time', lambda x: x.quantile(0.90)),
            median_delay=('Delivery_Time', 'median'),
        ).reset_index()

        grouped = grouped[grouped['n'] >= MIN_SAMPLE].copy()
        grouped['dimension'] = dim_name
        grouped['segment_label'] = grouped[cols].astype(str).agg(' + '.join, axis=1)
        grouped['breach_lift_pct'] = (grouped['breach_rate'] - overall_breach) / overall_breach * 100
        grouped['delay_diff_min'] = grouped['avg_delay'] - overall_delay
        grouped['suggested_buffer_pct'] = ((grouped['p90_delay'] - grouped['median_delay']) / grouped['median_delay'] * 100).round(0)
        grouped['suggested_buffer_min'] = (grouped['p90_delay'] - grouped['median_delay']).round(0)

        candidates.append(grouped[['dimension', 'segment_label', 'n', 'breach_rate', 'breach_lift_pct',
                                    'avg_delay', 'delay_diff_min', 'suggested_buffer_pct', 'suggested_buffer_min']])

    all_candidates = pd.concat(candidates, ignore_index=True)

    findings = all_candidates[all_candidates['breach_lift_pct'].abs() >= 15].copy()
    findings = findings.sort_values('breach_lift_pct', ascending=False)

    findings['n_norm'] = (findings['n'] - findings['n'].min()) / (findings['n'].max() - findings['n'].min())
    findings['lift_norm'] = (findings['breach_lift_pct'].abs() - findings['breach_lift_pct'].abs().min()) / \
                             (findings['breach_lift_pct'].abs().max() - findings['breach_lift_pct'].abs().min())

    findings['score_customer_satisfaction'] = (0.75 * findings['lift_norm'] + 0.25 * findings['n_norm']).round(3)
    findings['score_cost_reduction'] = (0.35 * findings['lift_norm'] + 0.65 * findings['n_norm']).round(3)

    findings.to_pickle(output_path)

    if verbose:
        print(f"Overall breach rate: {overall_breach*100:.1f}% | Overall avg delay: {overall_delay:.1f} min")
        print(f"Segments scanned: {len(all_candidates)} | Meeting min sample ({MIN_SAMPLE}): {len(all_candidates)}")
        print(f"Meaningful findings (|lift| >= 15%): {len(findings)}")

    return findings


if __name__ == '__main__':
    pd.set_option('display.max_colwidth', 40)
    pd.set_option('display.width', 160)

    findings = run_insight_engine()
    print()
    print("=== TOP 8 BY CUSTOMER SATISFACTION SCORE ===")
    cols = ['dimension','segment_label','n','breach_rate','breach_lift_pct','suggested_buffer_pct','score_customer_satisfaction']
    print(findings.sort_values('score_customer_satisfaction', ascending=False)[cols].head(8).to_string(index=False))
    print()
    print("=== TOP 8 BY COST REDUCTION SCORE ===")
    cols2 = ['dimension','segment_label','n','breach_rate','breach_lift_pct','suggested_buffer_pct','score_cost_reduction']
    print(findings.sort_values('score_cost_reduction', ascending=False)[cols2].head(8).to_string(index=False))
