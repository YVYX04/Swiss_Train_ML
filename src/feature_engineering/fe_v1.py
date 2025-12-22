"""
                    +----------------------------------+
                    | Features Engineering on Data IST |
                    +----------------------------------+

                    Copyright © 2025, 2026 Yvan Richard
                    All rights reserved.

This source code is used to create new features from the cleaned IST dataset,
to be used in my machine learning models.

"""

import os
import pandas as pd
import glob
from pathlib import Path
from datetime import datetime, timedelta

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
        linie_geo_df[['stop_id', 'latitude', 'longitude']],
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

    # save feature data
    out_name = f"ist_features_{month}.parquet"
    out_path = feature_dir / out_name
    df.to_parquet(out_path, index=False)
    print(f"Saved feature data to: {out_path} with {len(df):,} rows")

if __name__ == "__main__":
    for month in MONTHS:
        process_month(month)



    







 



 