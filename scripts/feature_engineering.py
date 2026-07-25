"""
Feature engineering.

Refactored in the Render-deployment fix into a callable function
(run_feature_engineering) so api/app.py can call it in-process instead of
via subprocess. Running it as a subprocess meant a second, separate Python
interpreter reloading pandas/numpy on every refresh, on Render's free
512MB tier, that pushed memory usage over the limit and triggered a worker
timeout + SIGKILL. Calling it directly in the same process removes that
duplicated memory cost. The standalone script behaviour (running this file
directly, printing the summary report) is unchanged, kept in the
__main__ block below.
"""
import pandas as pd
import numpy as np


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def hour_band(h):
    if pd.isna(h): return np.nan
    if 5 <= h < 11: return 'Morning'
    if 11 <= h < 16: return 'Midday'
    if 16 <= h < 21: return 'Evening'
    return 'Night'


def run_feature_engineering(csv_path='../data/raw/amazon_delivery.csv',
                             output_path='../data/processed/cleaned_data.pkl',
                             verbose=True):
    df = pd.read_csv(csv_path)
    n_raw = len(df)

    str_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].isin(['nan', 'NaN', '']), c] = np.nan

    df['Area'] = df['Area'].replace({'Metropolitian': 'Metropolitan'})

    before = len(df)
    df = df.dropna(subset=['Weather', 'Traffic', 'Agent_Rating']).reset_index(drop=True)
    dropped_nulls = before - len(df)

    bad_geocode = (df['Store_Latitude'].abs() < 1) & (df['Store_Longitude'].abs() < 1)
    df['Distance_km'] = np.where(
        bad_geocode, np.nan,
        haversine(df['Store_Latitude'], df['Store_Longitude'], df['Drop_Latitude'], df['Drop_Longitude'])
    )
    n_bad_geocode = bad_geocode.sum()

    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df['Weekday'] = df['Order_Date'].dt.day_name()
    df['Is_Weekend'] = df['Order_Date'].dt.dayofweek.isin([5, 6])

    holiday_dates = pd.to_datetime(['2022-03-01', '2022-03-18'])
    df['Is_Holiday'] = df['Order_Date'].isin(holiday_dates)

    df['Order_Hour'] = pd.to_datetime(df['Order_Time'], format='%H:%M:%S', errors='coerce').dt.hour
    df['Time_Band'] = df['Order_Hour'].apply(hour_band)

    df['SLA_Threshold'] = df.groupby('Area')['Delivery_Time'].transform(lambda x: x.quantile(0.75))
    df['SLA_Breach'] = df['Delivery_Time'] > df['SLA_Threshold']

    df.to_pickle(output_path)

    if verbose:
        print(f"Raw rows: {n_raw}")
        print(f"Dropped (null Weather/Traffic/Agent_Rating): {dropped_nulls}")
        print(f"Final row count: {len(df)}")
        print(f"Rows with bad geocode (Distance_km = null): {n_bad_geocode} ({n_bad_geocode/len(df)*100:.1f}%)")

    return df


if __name__ == '__main__':
    df = run_feature_engineering()
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
