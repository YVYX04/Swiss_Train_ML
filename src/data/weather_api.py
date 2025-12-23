import requests
import pandas as pd

# Based on my examination of the Swiss railway network, I have selected the following cities
# to have a representative sample of weather conditions across Switzerland.
CITIES = [
    "Zurich", "Geneva", "Martigny", "Neuchatel",
    "Lausanne", "Winterthur", "Luzern", "Basel", "Lugano"
]

PERIODS = [
    ("2025-01-01", "2025-01-31"),
    ("2025-09-01", "2025-09-30"),
]

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
]

def geocode_city(name: str) -> tuple[float, float, str]:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": name, "country": "CH", "count": 1, "language": "en", "format": "json"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data.get("results"):
        raise ValueError(f"Geocoding failed for: {name}")
    hit = data["results"][0]
    return hit["latitude"], hit["longitude"], hit["name"]

def fetch_daily(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Europe/Zurich",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame({"date": d["time"]})
    for v in DAILY_VARS:
        df[v] = d.get(v)
    return df

def main():
    rows = []
    for city in CITIES:
        lat, lon, canonical = geocode_city(city)
        for start, end in PERIODS:
            df = fetch_daily(lat, lon, start, end)
            df.insert(0, "city", canonical)
            df.insert(1, "latitude", lat)
            df.insert(2, "longitude", lon)
            df.insert(3, "period", f"{start}_to_{end}")
            rows.append(df)

    out = pd.concat(rows, ignore_index=True).sort_values(["city", "date"])
    out.to_csv("../../data/weather/swiss_cities_weather_daily_jan_sep_2025.csv", index=False)
    print("Saved: ../../data/weather/swiss_cities_weather_daily_jan_sep_2025.csv")

if __name__ == "__main__":
    main()