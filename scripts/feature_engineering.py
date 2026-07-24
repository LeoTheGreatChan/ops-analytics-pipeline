import pandas as pd
import numpy as np

# ---- 1. LOAD ----
df = pd.read_csv('../data/raw/amazon_delivery.csv')
n_raw = len(df)

# ---- 2. CLEAN ----
str_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
for c in str_cols:
    df[c] = df[c].astype(str).str.strip()
    df.loc[df[c].isin(['nan', 'NaN', '']), c] = np.nan

# fix source typo
df['Area'] = df['Area'].replace({'Metropolitian': 'Metropolitan'})

# drop rows with nulls in key categorical fields (small: Weather 91, Traffic ~some, Agent_Rating 54)
before = len(df)
df = df.dropna(subset=['Weather', 'Traffic', 'Agent_Rating']).reset_index(drop=True)
dropped_nulls = before - len(df)

# ---- 3. DISTANCE (haversine), null out bad geocodes ----
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

bad_geocode = (df['Store_Latitude'].abs() < 1) & (df['Store_Longitude'].abs() < 1)
df['Distance_km'] = np.where(
    bad_geocode, np.nan,
    haversine(df['Store_Latitude'], df['Store_Longitude'], df['Drop_Latitude'], df['Drop_Longitude'])
)
n_bad_geocode = bad_geocode.sum()

# ---- 4. DATE / TIME FEATURES ----
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['Weekday'] = df['Order_Date'].dt.day_name()
df['Is_Weekend'] = df['Order_Date'].dt.dayofweek.isin([5, 6])

# India public holidays falling within the dataset's date window (11 Feb - 6 Apr 2022)
# Confirmed via search: Maha Shivaratri (1 Mar 2022), Holi (18 Mar 2022)
holiday_dates = pd.to_datetime(['2022-03-01', '2022-03-18'])
df['Is_Holiday'] = df['Order_Date'].isin(holiday_dates)

# Order hour -> time-of-day band
df['Order_Hour'] = pd.to_datetime(df['Order_Time'], format='%H:%M:%S', errors='coerce').dt.hour

def hour_band(h):
    if pd.isna(h): return np.nan
    if 5 <= h < 11: return 'Morning'
    if 11 <= h < 16: return 'Midday'
    if 16 <= h < 21: return 'Evening'
    return 'Night'

df['Time_Band'] = df['Order_Hour'].apply(hour_band)

# ---- 5. SLA BREACH FLAG (relative threshold: 75th percentile per Area) ----
df['SLA_Threshold'] = df.groupby('Area')['Delivery_Time'].transform(lambda x: x.quantile(0.75))
df['SLA_Breach'] = df['Delivery_Time'] > df['SLA_Threshold']

# ---- 6. SAVE ----
df.to_pickle('../data/processed/cleaned_data.pkl')

# ---- 7. SUMMARY REPORT ----
print(f"Raw rows: {n_raw}")
print(f"Dropped (null Weather/Traffic/Agent_Rating): {dropped_nulls}")
print(f"Final row count: {len(df)}")
print(f"Rows with bad geocode (Distance_km = null): {n_bad_geocode} ({n_bad_geocode/len(df)*100:.1f}%)")
print(f"Date range: {df['Order_Date'].min().date()} to {df['Order_Date'].max().date()} ({(df['Order_Date'].max()-df['Order_Date'].min()).days} days)")
print(f"Holiday-flagged rows: {df['Is_Holiday'].sum()} ({df['Is_Holiday'].sum()/len(df)*100:.2f}%)")
print()
print("Overall SLA breach rate:", f"{df['SLA_Breach'].mean()*100:.1f}%")
print()
print("Breach rate by Area:")
print((df.groupby('Area')['SLA_Breach'].mean()*100).round(1))
print()
print("Breach rate by Weather:")
print((df.groupby('Weather')['SLA_Breach'].mean()*100).round(1).sort_values(ascending=False))
print()
print("Breach rate by Traffic:")
print((df.groupby('Traffic')['SLA_Breach'].mean()*100).round(1).sort_values(ascending=False))
print()
print("Breach rate by Weekday:")
print((df.groupby('Weekday')['SLA_Breach'].mean()*100).round(1).reindex(
    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']))
print()
print("Holiday vs non-holiday breach rate:")
print((df.groupby('Is_Holiday')['SLA_Breach'].mean()*100).round(1))
print()
print("Rows per weekday x holiday (checking sample size):")
print(df.groupby(['Weekday','Is_Holiday']).size().unstack(fill_value=0))
