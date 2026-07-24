import pandas as pd
import numpy as np
import json
import pickle

df = pd.read_pickle('../data/processed/cleaned_data.pkl')
with open('../data/processed/model.pkl','rb') as f:
    model_data = pickle.load(f)

OVERALL_BREACH = df['SLA_Breach'].mean()
OVERALL_DELAY = df['Delivery_Time'].mean()

# ---- 1. KPIs ----
kpis = {
    "on_time_rate": round((1 - OVERALL_BREACH) * 100, 1),
    "avg_delay_min": round(OVERALL_DELAY, 1),
    "sla_breach_rate": round(OVERALL_BREACH * 100, 1),
    "total_deliveries": len(df),
    "date_range": f"{df['Order_Date'].min().date()} to {df['Order_Date'].max().date()}",
    "model_accuracy": round(model_data['accuracy'] * 100, 1),
    "model_f1_breach_class": round(model_data['f1'], 3)
}

# breach rate by area (for Overview bar chart)
area_breach = (df.groupby('Area')['SLA_Breach'].mean() * 100).round(1).to_dict()
kpis['breach_by_area'] = area_breach

# weekly trend (real 8-week window)
df['week'] = df['Order_Date'].dt.isocalendar().week
weekly = (df.groupby('week')['SLA_Breach'].mean() * 100).round(1)
kpis['weekly_breach_trend'] = weekly.tolist()

with open('../data/processed/kpis.json','w') as f:
    json.dump(kpis, f, indent=2)

# ---- 2. SEGMENTS ----
weather_breach = (df.groupby('Weather')['SLA_Breach'].mean() * 100).round(1).sort_values(ascending=False).to_dict()

def risk_tier(rate):
    if rate > 30: return 'High'
    if rate < 12: return 'Low'
    return 'Moderate'

seg_table = df.groupby(['Area','Traffic']).agg(
    n=('SLA_Breach','size'), avg_delay=('Delivery_Time','mean'), breach_rate=('SLA_Breach','mean')
).reset_index()
seg_table = seg_table[seg_table['n'] >= 200].sort_values('breach_rate', ascending=False).head(10)
seg_table['avg_delay'] = seg_table['avg_delay'].round(0).astype(int)
seg_table['breach_rate'] = (seg_table['breach_rate']*100).round(1)
seg_table['risk'] = seg_table['breach_rate'].apply(risk_tier)
seg_table['segment'] = seg_table['Area'] + ' · ' + seg_table['Traffic'] + ' traffic'

segments = {
    "weather_breach_rate": weather_breach,
    "segment_table": seg_table[['segment','n','avg_delay','breach_rate','risk']].to_dict('records')
}
with open('../data/processed/segments.json','w') as f:
    json.dump(segments, f, indent=2)

# ---- 3. FORECAST (weekday pattern, with honest holiday caveat) ----
weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
weekday_breach = (df.groupby('Weekday')['SLA_Breach'].mean() * 100).round(1).reindex(weekday_order).to_dict()

holiday_breach = (df.groupby('Is_Holiday')['SLA_Breach'].mean() * 100).round(1).to_dict()
n_holidays = df['Is_Holiday'].sum()
holiday_dates_str = "1 March 2022 (Maha Shivaratri), 18 March 2022 (Holi)"

forecast = {
    "weekday_breach_rate": weekday_breach,
    "holiday_vs_non_holiday": {"non_holiday": holiday_breach.get(False), "holiday": holiday_breach.get(True)},
    "holiday_note": f"Only 2 public holidays fall within this dataset's {kpis['date_range']} window ({holiday_dates_str}), landing on Tuesdays and Fridays only. Holiday effect ({holiday_breach.get(True)}% vs {holiday_breach.get(False)}% breach rate) is directional, not statistically robust given the small sample ({n_holidays} holiday-flagged rows).",
    "highest_risk_day": max(weekday_breach, key=weekday_breach.get),
    "lowest_risk_day": min(weekday_breach, key=weekday_breach.get)
}
with open('../data/processed/forecast.json','w') as f:
    json.dump(forecast, f, indent=2)

print("KPIs:", json.dumps(kpis, indent=2))
print()
print("Segments (weather):", json.dumps(segments['weather_breach_rate'], indent=2))
print()
print("Forecast:", json.dumps(forecast, indent=2))
