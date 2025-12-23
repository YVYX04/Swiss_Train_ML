"""
                    +----------------------------------+
                    | Features Engineering on Data IST |
                    +----------------------------------+

                    Copyright © 2025, 2026 Yvan Richard
                    All rights reserved.

This source code is used to create new features from the cleaned IST dataset,
to be used in my machine learning models.

+ weather data integration

"""

import os
import pandas as pd
import glob
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# ---------
# CONFIG
# ---------

INTERIM_BASE_DIR = Path("../../data/interim")
FEATURES_BASE_DIR = Path("../../data/features")

# months
MONTHS = ["2025_01", "2025_09"]

# columns to keep from cleaned data (.parquet)
COLUMNS_TO_KEEP = [
    "op_date",
    "trip_id",
    "stop_id",
    "stop_name",
    "line_name",
    "vehicle_type",
    "additional_trip",
    "arrival_scheduled_dt",
    "arrival_observed_dt",
    "arrival_delay_minutes",
    "is_delayed"
]

# weather (daily) file produced by your Open-Meteo fetch script
# Expected columns: city, latitude, longitude, date, plus the daily variables below.
WEATHER_DAILY_PATH = Path("../../data/weather/swiss_cities_weather_daily_jan_sep_2025.csv")

# daily vars we want to merge (must exist as columns in WEATHER_DAILY_PATH)
WEATHER_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
]

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features from the data set. The new features are:
    - hour_of_day: Hour of the day (0-23) of the scheduled arrival time.
    - day_of_week: Day of the week (0=Monday, 6=Sunday) of the operation date.
    - is_weekend: Boolean indicating if the operation date is a weekend (Saturday or Sunday).
    - is_peak: Boolean indicating if the scheduled arrival time is during peak hours (7-9 AM and 4-6 PM).


    Args:
        df (pd.DataFrame): Input dataframe with pre-processed IST data.
    Returns:
        pd.DataFrame: Dataframe with new temporal features.
    """
    # hour of day
    df['hour_of_day'] = df['arrival_scheduled_dt'].dt.hour

    # day of week
    df['day_of_week'] = df['op_date'].dt.dayofweek

    # is weekend
    df['is_weekend'] = df['day_of_week'].isin([5, 6])

    # is peak hours (7-9 AM and 4-6 PM)
    df['is_peak'] = df['hour_of_day'].isin([7, 8, 16, 17, 18])

    return df

def create_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create network features from the data set. The new features are:
    - vt_[vehicle_type]: One-hot encoded columns for each vehicle type. (14 unique types)
    - stop_lat: Latitude of the stop (to be merged from linie_geo dataset).
    - stop_lon: Longitude of the stop (to be merged from linie_geo dataset).
    - connection_density: Level of connectedness of the stop (number of unique lines serving the stop).
    """

    # vt_[vehicle_type]: One-hot encoding for vehicle_type
    vehicle_type_dummies = pd.get_dummies(df['vehicle_type'], prefix='vt')
    df = pd.concat([df, vehicle_type_dummies], axis=1)

    # stop_lat, stop_lon: These are the precise latitude and longitude of each stop.
    # I have to rely on another dataset provided by SBB at:
    # https://data.sbb.ch/explore/dataset/linie-mit-betriebspunkten/export/

    linie_geo_path = INTERIM_BASE_DIR / "linie_geo_clean.parquet"
    linie_geo_df = pd.read_parquet(linie_geo_path)

    # merge lat/lon from linie_geo dataset
    # first convert stop_id to numeric to match types
    df['stop_id'] = pd.to_numeric(df['stop_id'], errors='coerce')
    linie_geo_df['stop_id'] = pd.to_numeric(linie_geo_df['stop_id'], errors='coerce')

    # IMPORTANT: linie_geo can contain multiple rows per stop_id (e.g., one stop served by many lines).
    # If we merge without deduplicating, we create a one-to-many join that duplicates IST rows and
    # breaks downstream rolling features (same event appears multiple times).
    linie_geo_df = (
        linie_geo_df
        .dropna(subset=['stop_id'])
        .sort_values(['stop_id'])
        .drop_duplicates(subset=['stop_id'], keep='first')
    )

    df = df.merge(
        linie_geo_df[['stop_id', 'latitude', 'longitude']].rename(
            columns={'latitude': 'stop_lat', 'longitude': 'stop_lon'}
        ),
        on='stop_id',
        how='left',
        validate='m:1'  # many IST rows -> one geo row per stop_id
    )

    # connection_density: Number of unique lines serving the stop
    df['connection_density'] = df.groupby('stop_name')['trip_id'].transform('nunique')

    return df

def create_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    create_congestion_features adds features related to congestion patterns based on historical delay data.
    Args:
        df (pd.DataFrame): Input dataframe with pre-processed IST data.
    Returns:
        pd.DataFrame: Dataframe with new congestion features.

    Warning:
        - time delay & no data leakage: I use .shift(1) to avoid "instatenous data access"
        kind of leakage

    Ideas:
        - running_trip_delay: mean delay on the trip past stops (3 stops window)
        - mean_stop_delay: mean delay at the stop name 
    """
    # Pandas aligns on the index when assigning a Series back to the DataFrame.
    # If the DataFrame index contains duplicates (can happen after merges / filtering),
    # the assignment below will fail with: "cannot reindex on an axis with duplicate labels".
    # I have to ensure a unique index before creating rolling features.
    df = df.reset_index(drop=True)

    # Rolling with a time-based window requires a proper datetime column.

    # running_trip_delay
    # For time-based rolling, each group must be monotonic in the 'on' column.
    df = df.sort_values(["trip_id", "arrival_scheduled_dt"], kind="mergesort").reset_index(drop=True)
    df['running_trip_delay'] = (
        df.groupby('trip_id')['arrival_delay_minutes']
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    )

    # mean_stop_delay (time-windowed): mean delay at the stop over the past 90 minutes
    # Important: the rolling output can have duplicate index labels (e.g., duplicate timestamps).
    # Assign by position (numpy) to avoid fragile index-based alignment.
    df = df.sort_values(["stop_name", "arrival_scheduled_dt"], kind="mergesort").reset_index(drop=True)

    tmp_stop = (
        df
        .groupby("stop_name")
        .rolling(
            window="90min",
            on="arrival_scheduled_dt",
            closed="left"  # strictly past info: [t-90min, t)
        )["arrival_delay_minutes"]
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["mean_stop_delay"] = tmp_stop.to_numpy()
    print("Created congestion features: running_trip_delay, mean_stop_delay\n")
    

    return df


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance (km). Supports numpy broadcasting."""
    R = 6371.0
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge (i) national daily indicators and (ii) local daily weather by nearest city-center.

    Requirements:
    - df must contain: op_date (datetime), stop_id, stop_lat, stop_lon
    - WEATHER_DAILY_PATH must exist

    Output columns added:
    - date (helper join key)
    - nat_* : national daily indicators (mean over all cities)
    - nat_sum_* : national daily sums for precipitation/rain/snowfall
    - wx_city : nearest weather center city for each stop_id
    - wx_* : daily weather vars for that wx_city on that date
    - v_g_norm : L2 norm of your v_g vector (defined by VG_VARS)
    """

    if not WEATHER_DAILY_PATH.exists():
        raise FileNotFoundError(
            f"Weather file not found at {WEATHER_DAILY_PATH}. "
            "Place the CSV there (or change WEATHER_DAILY_PATH)."
        )

    # Load weather
    weather = pd.read_csv(WEATHER_DAILY_PATH)
    if "date" not in weather.columns:
        raise ValueError("Weather CSV must contain a 'date' column.")

    weather["date"] = pd.to_datetime(weather["date"], errors="raise").dt.date

    # Keep only needed columns (fail fast if missing)
    required_cols = {"city", "latitude", "longitude", "date"}.union(WEATHER_DAILY_VARS)
    missing = required_cols.difference(weather.columns)
    if missing:
        raise ValueError(f"Weather CSV is missing columns: {sorted(missing)}")

    weather = weather[["city", "latitude", "longitude", "date"] + WEATHER_DAILY_VARS].copy()

    print(f"Loaded weather data with {len(weather):,} rows for {weather['city'].nunique():,} cities.")

    # 1) National indicators (aggregated across cities per day)
    nat = (
        weather
        .groupby("date", as_index=False)[WEATHER_DAILY_VARS]
        .mean()
        .rename(columns={c: f"nat_{c}" for c in WEATHER_DAILY_VARS})
    )

    # Add national sums for precipitation-like variables (optional but usually useful)
    sum_base = [c for c in ["precipitation_sum", "rain_sum", "snowfall_sum"] if c in WEATHER_DAILY_VARS]
    if sum_base:
        nat_sum = (
            weather
            .groupby("date", as_index=False)[sum_base]
            .sum()
            .rename(columns={c: f"nat_sum_{c}" for c in sum_base})
        )
        nat = nat.merge(nat_sum, on="date", how="left")

    # Join national indicators on date
    df = df.copy()
    df["date"] = df["op_date"].dt.date
    df = df.merge(nat, on="date", how="left", validate="m:1")

    print(f"Joined national weather indicators on date. New columns: {list(nat.columns)}")
    # print first rows of df
    print(df.head(2))

    # 2) Map each stop_id to its nearest weather center (city)
    stops = (
        df[["stop_id", "stop_lat", "stop_lon"]]
        .dropna(subset=["stop_id", "stop_lat", "stop_lon"])
        .drop_duplicates(subset=["stop_id"])
        .copy()
    )

    centers = (
        weather[["city", "latitude", "longitude"]]
        .drop_duplicates(subset=["city"])
        .reset_index(drop=True)
    )

    # Distance matrix: (#stops x #centers)
    s_lat = stops["stop_lat"].to_numpy()[:, None]
    s_lon = stops["stop_lon"].to_numpy()[:, None]
    c_lat = centers["latitude"].to_numpy()[None, :]
    c_lon = centers["longitude"].to_numpy()[None, :]

    dist_km = haversine_km(s_lat, s_lon, c_lat, c_lon)
    nearest_idx = dist_km.argmin(axis=1)

    stops["wx_city"] = centers.loc[nearest_idx, "city"].to_numpy()

    # Join wx_city back
    df = df.merge(stops[["stop_id", "wx_city"]], on="stop_id", how="left", validate="m:1")

    # 3) Join local weather (by date + wx_city)
    wx = weather.copy()
    wx = wx.rename(columns={"city": "wx_city"})
    wx = wx.rename(columns={c: f"wx_{c}" for c in WEATHER_DAILY_VARS})

    df = df.merge(
        wx[["date", "wx_city"] + [f"wx_{c}" for c in WEATHER_DAILY_VARS]],
        on=["date", "wx_city"],
        how="left",
        validate="m:1"
    )

    print(f"Joined local weather by wx_city and date. New columns: {[f'wx_{c}' for c in WEATHER_DAILY_VARS]}")

    return df


# ---------
# MAIN
# ---------
"""
Due to the congestion features depending on historical data, I need to process
the data month by month, in chronological order.
"""

def process_month(month: str):
    """Process a single month of data to create features.

    Args:
        month (str): Month in the format 'YYYY_MM'.
    """
    print(f"Processing month: {month}")

    interim_dir = INTERIM_BASE_DIR
    feature_dir = FEATURES_BASE_DIR
    feature_dir.mkdir(parents=True, exist_ok=True)

    # read cleaned data for the month
    file_path = interim_dir / f"ist_clean_{month}.parquet"
    df = pd.read_parquet(file_path)

    # keep only relevant columns
    df = df[COLUMNS_TO_KEEP].copy()

    # make sure datetime columns are in datetime format
    df['arrival_scheduled_dt'] = pd.to_datetime(df['arrival_scheduled_dt'], errors='coerce')
    df['arrival_observed_dt'] = pd.to_datetime(df['arrival_observed_dt'], errors='coerce')
    df['op_date'] = pd.to_datetime(df['op_date'], format="%d.%m.%Y", errors='raise')

    # create temporal features
    df = create_temporal_features(df)

    # create network features
    df = create_network_features(df)

    # create congestion features
    df = create_congestion_features(df)

    # create weather features
    df = create_weather_features(df)

    # save feature data
    out_name = f"ist_features_weather_{month}.parquet"
    out_path = feature_dir / out_name
    df.to_parquet(out_path, index=False)
    print(f"Saved feature data to: {out_path} with {len(df):,} rows")

if __name__ == "__main__":
    for month in MONTHS:
        process_month(month)

