import pandas as pd
import numpy as np
import json
import pickle

df = pd.read_pickle('../data/processed/cleaned_data.pkl')
findings = pd.read_pickle('../data/processed/findings.pkl')
with open('../data/processed/model.pkl','rb') as f:
    model_data = pickle.load(f)

OVERALL_BREACH = df['SLA_Breach'].mean()
OVERALL_DELAY = df['Delivery_Time'].mean()

# ---- ADD AGENT_RATING AS A FINDING (numeric feature, bucketed) ----
df['Agent_Rating_Bucket'] = pd.cut(df['Agent_Rating'], bins=[0,3,4,4.5,4.8,5.1],
                                     labels=['Below 3.0','3.0-4.0','4.0-4.5','4.5-4.8','4.8-5.0'])
rating_grp = df.groupby('Agent_Rating_Bucket', observed=True).agg(
    n=('SLA_Breach','size'), breach_rate=('SLA_Breach','mean'),
    avg_delay=('Delivery_Time','mean'), p90_delay=('Delivery_Time', lambda x: x.quantile(0.9)),
    median_delay=('Delivery_Time','median')
).reset_index()
rating_grp = rating_grp[rating_grp['n'] >= 200].copy()
rating_grp['dimension'] = 'Agent_Rating'
rating_grp['segment_label'] = rating_grp['Agent_Rating_Bucket'].astype(str)
rating_grp['breach_lift_pct'] = (rating_grp['breach_rate'] - OVERALL_BREACH) / OVERALL_BREACH * 100
rating_grp['delay_diff_min'] = rating_grp['avg_delay'] - OVERALL_DELAY
rating_grp['suggested_buffer_pct'] = ((rating_grp['p90_delay'] - rating_grp['median_delay']) / rating_grp['median_delay'] * 100).round(0)
rating_grp['suggested_buffer_min'] = (rating_grp['p90_delay'] - rating_grp['median_delay']).round(0)
rating_grp = rating_grp[['dimension','segment_label','n','breach_rate','breach_lift_pct','avg_delay','delay_diff_min','suggested_buffer_pct','suggested_buffer_min']]
rating_findings = rating_grp[rating_grp['breach_lift_pct'].abs() >= 15].copy()

all_findings = pd.concat([findings, rating_findings], ignore_index=True)
n_norm = (all_findings['n'] - all_findings['n'].min()) / (all_findings['n'].max() - all_findings['n'].min())
lift_norm = (all_findings['breach_lift_pct'].abs() - all_findings['breach_lift_pct'].abs().min()) / \
            (all_findings['breach_lift_pct'].abs().max() - all_findings['breach_lift_pct'].abs().min())
all_findings['score_customer_satisfaction'] = (0.75*lift_norm + 0.25*n_norm).round(3)
all_findings['score_cost_reduction'] = (0.35*lift_norm + 0.65*n_norm).round(3)

# ---- WRITE-UP FUNCTION (this is the step an LLM call performs in production;
# written here directly since this model IS the LLM that would generate it) ----
def write_insight(row):
    seg = row['segment_label']
    dim = row['dimension']
    lift = row['breach_lift_pct']
    rate = row['breach_rate']*100
    buf = row['suggested_buffer_pct']
    n = int(row['n'])
    direction = "higher" if lift > 0 else "lower"

    if dim == 'Weather x Traffic':
        w, t = seg.split(' + ')
        finding = f"{w} weather combined with {t} traffic shows a **{abs(lift):.0f}% {direction}** breach rate than average ({rate:.1f}% vs {OVERALL_BREACH*100:.1f}% baseline)."
        action = f"Add a **{buf:.0f}% buffer** to delivery estimates for this combination." if lift > 0 else "No buffer adjustment needed; this combination consistently outperforms baseline."
    elif dim == 'Area x Traffic':
        a, t = seg.split(' + ')
        finding = f"{a} deliveries during {t} traffic show a **{abs(lift):.0f}% {direction}** breach rate than average ({rate:.1f}% vs {OVERALL_BREACH*100:.1f}% baseline)."
        action = f"Add a **{buf:.0f}% buffer** for routes in this segment." if lift > 0 else "Standard SLA buffer is sufficient for this segment."
    elif dim == 'Agent_Rating':
        finding = f"Agents rated **{seg}** show a **{abs(lift):.0f}% {direction}** breach rate than average ({rate:.1f}% vs {OVERALL_BREACH*100:.1f}% baseline)."
        action = "Prioritise training or route reassignment for lower-rated agents." if lift > 0 else "No action needed; this rating band performs at or above target."
    else:
        finding = f"{dim} = {seg} shows a **{abs(lift):.0f}% {direction}** breach rate than average ({rate:.1f}% vs {OVERALL_BREACH*100:.1f}% baseline)."
        action = f"Add a **{buf:.0f}% buffer** for this segment." if lift > 0 else "This segment consistently outperforms baseline; no action needed."

    impact_pp = abs(rate - OVERALL_BREACH*100)
    return finding + " " + action, f"estimated impact: {'-' if lift>0 else '+'}{impact_pp:.0f}pp breach rate · {n:,} deliveries/period affected"

for obj_col, obj_name in [('score_customer_satisfaction','customer_satisfaction'), ('score_cost_reduction','cost_reduction')]:
    ranked = all_findings.sort_values(obj_col, ascending=False).head(6).reset_index(drop=True)
    out = []
    for i, row in ranked.iterrows():
        finding_text, impact_text = write_insight(row)
        out.append({
            "rank": i+1,
            "top_priority": i == 0,
            "dimension": row['dimension'],
            "segment": row['segment_label'],
            "finding": finding_text,
            "impact": impact_text,
            "n": int(row['n']),
            "breach_rate": round(row['breach_rate']*100,1),
            "breach_lift_pct": round(row['breach_lift_pct'],1)
        })
    with open(f'../data/processed/insights_{obj_name}.json','w') as f:
        json.dump(out, f, indent=2)
    print(f"=== {obj_name.upper()} — top 3 preview ===")
    for item in out[:3]:
        print(f"#{item['rank']} {'[TOP PRIORITY] ' if item['top_priority'] else ''}{item['finding']}")
        print(f"   {item['impact']}")
    print()

print("Full insight JSON files written for both objectives.")
