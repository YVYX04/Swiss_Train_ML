"""
                                        +----------------+
                                        | Clean Data IST |
                                        +----------------+

                                Copyright © 2025, 2026 Yvan Richard
                                All rights reserved.

This source code is used to clean and preprocess data from the IST dataset,
made available from the website swiss.opendata.ch.

"""

import os
import glob
import pandas as pd
from pathlib import Path

# ---------
# CONFIG
# ---------

RAW_BASE_DIR = Path("../../data/raw")
INTERIM_BASE_DIR = Path("../../data/interim")

# the months
MONTHS = ["2025_01", "2025_09"]

# Raw column names from IST docs
USECOLS = [
    "BETRIEBSTAG",
    "FAHRT_BEZEICHNER",
    "BETREIBER_ABK",
    "PRODUKT_ID",
    "LINIEN_ID",
    "LINIEN_TEXT",
    "UMLAUF_ID",
    "VERKEHRSMITTEL_TEXT",
    "ZUSATZFAHRT_TF",
    "FAELLT_AUS_TF",
    "BPUIC",
    "HALTESTELLEN_NAME",
    "ANKUNFTSZEIT",
    "AN_PROGNOSE",
    "AN_PROGNOSE_STATUS",
    "ABFAHRTSZEIT",
    "AB_PROGNOSE",
    "AB_PROGNOSE_STATUS",
    "DURCHFAHRT_TF"#,
    # "SLOID",
]

COLUMNS_MAPPING = {
    "BETRIEBSTAG": "op_date",
    "FAHRT_BEZEICHNER": "trip_id",
    "BETREIBER_ABK": "operator_code",
    "PRODUKT_ID": "transport_type",
    "LINIEN_ID": "line_id",
    "LINIEN_TEXT": "line_name",
    "UMLAUF_ID": "route_id",
    "VERKEHRSMITTEL_TEXT": "vehicle_type",
    "ZUSATZFAHRT_TF": "additional_trip",
    "FAELLT_AUS_TF": "canceled_trip",
    "BPUIC": "stop_id",
    "HALTESTELLEN_NAME": "stop_name",
    "ANKUNFTSZEIT": "arrival_scheduled_raw",
    "AN_PROGNOSE": "arrival_observed_raw",
    "AN_PROGNOSE_STATUS": "arrival_status",
    "ABFAHRTSZEIT": "departure_scheduled_raw",
    "AB_PROGNOSE": "departure_observed_raw",
    "AB_PROGNOSE_STATUS": "departure_status",
    "DURCHFAHRT_TF": "pass_through"#,
    # "SLOID": "slo_id",
}


# delay thresholds in minutes
# Delay (min) = AN_PROGNOSE (when status = REAL) − ANKUNFTSZEIT
DELAY_THRESHOLD_MIN = 3  # "delayed" if arrival delay >= 3 minutes

# chunksize for processing
CHUNKSIZE = 200000  # number of rows per chunk when processing large files

# droppings
"""
	•	drop cancelled journeys: FAELLT_AUS_TF == true (the journey is canceled).
	•	drop pass-through stops: DURCHFAHRT_TF == true (the train doesn't halt, so delay is not meaningful).
"""

#-------------
# FUNCTIONS
#-------------
def process_file(file_path: Path, interim_dir: Path):
    print(f"Processing file: {file_path}")
    dfs = []

    for chunk in pd.read_csv(
        file_path,
        sep=";",
        usecols=USECOLS,
        dtype=str,
        chunksize=CHUNKSIZE,
    ):
        # rename columns
        chunk.rename(columns=COLUMNS_MAPPING, inplace=True)

        # basic cleaning: drop cancelled and pass-through
        # (null is treated as "false")
        chunk = chunk[
            (chunk["canceled_trip"] != "true")
            & (chunk["pass_through"] != "true")
        ].copy()

        # keep only rows where we have a target and an observed arrival
        chunk = chunk[
            chunk["arrival_scheduled_raw"].notna()
            & chunk["arrival_observed_raw"].notna()
        ].copy()
        if chunk.empty:
            continue

        # keep only REAL observed arrivals: effective actual arrival time
        chunk = chunk[chunk["arrival_status"] == "REAL"].copy()
        if chunk.empty:
            continue

        # parse datetimes (DD.MM.YYYY HH:MM and DD.MM.YYYY HH:MM:SS, seconds may be missing)
        chunk["arrival_scheduled_dt"] = pd.to_datetime(
            chunk["arrival_scheduled_raw"],
            dayfirst=True,
            errors="coerce",
        )
        chunk["arrival_observed_dt"] = pd.to_datetime(
            chunk["arrival_observed_raw"],
            dayfirst=True,
            errors="coerce",
        )

        # drop rows where parsing failed
        chunk = chunk[
            chunk["arrival_scheduled_dt"].notna()
            & chunk["arrival_observed_dt"].notna()
        ].copy()
        if chunk.empty:
            continue

        # compute delay in minutes
        delay = (
            chunk["arrival_observed_dt"] - chunk["arrival_scheduled_dt"]
        ).dt.total_seconds() / 60.0
        chunk["arrival_delay_minutes"] = delay

        # binary label
        chunk["is_delayed"] = (chunk["arrival_delay_minutes"] >= DELAY_THRESHOLD_MIN).astype("int8")

        # optional: drop crazy outliers (e.g. < -60 min or > 180 min)
        # chunk = chunk[
        #     (chunk["arrival_delay_minutes"] > -60)
        #     & (chunk["arrival_delay_minutes"] < 180)
        # ]

        if chunk.empty:
            continue

        # deduplicate per trip + stop (keep last actual arrival)
        chunk = chunk.sort_values("arrival_observed_dt")
        chunk = chunk.drop_duplicates(
            subset=["trip_id", "stop_id"], keep="last"
        )

        # WARNING: I focus only on SBB operator !
        # keep only rows where operator_code == SBB
        chunk = chunk[chunk["operator_code"] == "SBB"].copy()
        if chunk.empty:
            continue


        # keep only columns we need going forward
        keep_cols = [
            "op_date",
            "trip_id",
            "stop_id",
            "stop_name",
            "operator_id",
            "operator_code",
            "operator_name",
            "transport_type",
            "line_id",
            "line_name",
            "vehicle_type",
            "additional_trip",
            "arrival_scheduled_dt",
            "arrival_observed_dt",
            "arrival_delay_minutes",
            "is_delayed",
            "slo_id",
        ]
        keep_cols = [c for c in keep_cols if c in chunk.columns]
        chunk = chunk[keep_cols].copy()

        dfs.append(chunk)

    if not dfs:
        print(f"No valid rows in {file_path}, skipping.")
        return

    df_day = pd.concat(dfs, ignore_index=True)

    # infer a day label from BETRIEBSTAG or filename (simple version: use filename stem)
    out_name = Path(file_path).stem + ".parquet"
    out_path = interim_dir / out_name
    df_day.to_parquet(out_path, index=False)
    print(f"Saved cleaned data to: {out_path} with {len(df_day):,} rows")


def clean_ist_data():
    for month in MONTHS:
        raw_dir = RAW_BASE_DIR / month
        interim_dir = INTERIM_BASE_DIR / month
        os.makedirs(interim_dir, exist_ok=True)

        csv_files = sorted(glob.glob(str(raw_dir / "*.csv")))
        if not csv_files:
            print(f"No CSVs found in {raw_dir}")
            continue

        for file_path in csv_files:
            process_file(Path(file_path), interim_dir)

        # optional: merge all day files into a single month parquet
        month_files = sorted(glob.glob(str(interim_dir / "*.parquet")))
        if month_files:
            df_month = pd.concat(
                [pd.read_parquet(p) for p in month_files],
                ignore_index=True,
            )
            month_out = INTERIM_BASE_DIR / f"ist_clean_{month}.parquet"
            df_month.to_parquet(month_out, index=False)
            print(f"Saved month-level file {month_out} with {len(df_month):,} rows")


if __name__ == "__main__":
    print("Starting IST data cleaning...")
    clean_ist_data()

