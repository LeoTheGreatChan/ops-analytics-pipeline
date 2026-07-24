import pandas as pd
import numpy as np

df = pd.read_pickle('../data/processed/cleaned_data.pkl')

OVERALL_BREACH = df['SLA_Breach'].mean()
OVERALL_DELAY = df['Delivery_Time'].mean()
MIN_SAMPLE = 200  # segments below this are too thin to report confidently

# ---- 1. DEFINE SEGMENT DIMENSIONS TO SCAN ----
# single-dimension and a few meaningful compound dimensions
segment_defs = {
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

candidates = []

for dim_name, cols in segment_defs.items():
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

    # effect size: relative lift in breach rate vs overall baseline
    grouped['breach_lift_pct'] = (grouped['breach_rate'] - OVERALL_BREACH) / OVERALL_BREACH * 100
    grouped['delay_diff_min'] = grouped['avg_delay'] - OVERALL_DELAY

    # numeric buffer suggestion: p90 vs median gap, expressed as % of median
    grouped['suggested_buffer_pct'] = ((grouped['p90_delay'] - grouped['median_delay']) / grouped['median_delay'] * 100).round(0)
    grouped['suggested_buffer_min'] = (grouped['p90_delay'] - grouped['median_delay']).round(0)

    candidates.append(grouped[['dimension','segment_label','n','breach_rate','breach_lift_pct',
                                'avg_delay','delay_diff_min','suggested_buffer_pct','suggested_buffer_min']])

all_candidates = pd.concat(candidates, ignore_index=True)

# ---- 2. FILTER TO MEANINGFUL FINDINGS ----
# only keep segments with a real effect (breach lift beyond +/-15%) so we're not reporting noise
findings = all_candidates[all_candidates['breach_lift_pct'].abs() >= 15].copy()
findings = findings.sort_values('breach_lift_pct', ascending=False)

# ---- 3. BUSINESS-VALUE SCORING (two objectives) ----
# Customer satisfaction: weight breach severity + how bad the delay gets (p90 buffer) most heavily
# Cost reduction: weight volume affected (more deliveries = more driver/dispatch cost exposure) most heavily
findings['n_norm'] = (findings['n'] - findings['n'].min()) / (findings['n'].max() - findings['n'].min())
findings['lift_norm'] = (findings['breach_lift_pct'].abs() - findings['breach_lift_pct'].abs().min()) / \
                         (findings['breach_lift_pct'].abs().max() - findings['breach_lift_pct'].abs().min())

findings['score_customer_satisfaction'] = (0.75 * findings['lift_norm'] + 0.25 * findings['n_norm']).round(3)
findings['score_cost_reduction']       = (0.35 * findings['lift_norm'] + 0.65 * findings['n_norm']).round(3)

findings.to_pickle('../data/processed/findings.pkl')

# ---- 4. REPORT ----
pd.set_option('display.max_colwidth', 40)
pd.set_option('display.width', 160)

print(f"Overall breach rate: {OVERALL_BREACH*100:.1f}% | Overall avg delay: {OVERALL_DELAY:.1f} min")
print(f"Segments scanned: {len(all_candidates)} | Meeting min sample ({MIN_SAMPLE}): {len(all_candidates)}")
print(f"Meaningful findings (|lift| >= 15%): {len(findings)}")
print()
print("=== TOP 8 BY CUSTOMER SATISFACTION SCORE ===")
cols = ['dimension','segment_label','n','breach_rate','breach_lift_pct','suggested_buffer_pct','score_customer_satisfaction']
print(findings.sort_values('score_customer_satisfaction', ascending=False)[cols].head(8).to_string(index=False))
print()
print("=== TOP 8 BY COST REDUCTION SCORE ===")
cols2 = ['dimension','segment_label','n','breach_rate','breach_lift_pct','suggested_buffer_pct','score_cost_reduction']
print(findings.sort_values('score_cost_reduction', ascending=False)[cols2].head(8).to_string(index=False))
