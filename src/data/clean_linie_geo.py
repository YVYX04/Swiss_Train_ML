"""
                                        +----------------+
                                        | Clean linie   |
                                        +----------------+

                                Copyright © 2025, 2026 Yvan Richard
                                All rights reserved.

This source code is used to clean and preprocess data from an SBB dataset,
available at: https://data.sbb.ch/explore/dataset/linie-mit-betriebspunkten/export/


"""

import os
import glob
import pandas as pd
from pathlib import Path

# ---------
# CONFIG
# ---------


# ---------
# FUNCTIONS
# ---------

def clean_linie_geo(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the linie geo dataframe.

    Args:
        df (pd.DataFrame): Raw linie geo dataframe.
    Returns:
        pd.DataFrame: Cleaned linie geo dataframe.
    """
    # select relevant columns
    columns_to_keep = [
        "OPUIC",
        "Stop name",
        "Geopos"
    ]

    df_clean = df[columns_to_keep].copy()

    # decompose Geopos into latitude and longitude
    df_clean[['latitude', 'longitude']] = df_clean['Geopos'].str.split(',', expand=True)
    df_clean['latitude'] = pd.to_numeric(df_clean['latitude'], errors='coerce')
    df_clean['longitude'] = pd.to_numeric(df_clean['longitude'], errors='coerce')
    df_clean = df_clean.drop(columns=['Geopos'])

    # rename columns
    df_clean = df_clean.rename(columns={
        "OPUIC": "stop_id",
        "Stop name": "stop_name"
    })

    return df_clean





# ---------
# MAIN
# ---------
if __name__ == "__main__":
    # read raw linie geo data
    df_raw = pd.read_csv("../../data/raw/linie_geo.csv", sep=';')

    # clean data
    df_clean = clean_linie_geo(df_raw)

    # save cleaned data
    output_path = "../../data/interim/linie_geo_clean.parquet"
    df_clean.to_parquet(output_path, index=False)
    print(f"Saved cleaned linie geo data to: {output_path} with {len(df_clean):,} rows")




