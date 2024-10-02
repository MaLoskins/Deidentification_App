# utils/geocoding.py

import os
import re
import hashlib
import random
import string
import pandas as pd
import geopandas as gpd
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import spacy
import logging
import subprocess
import sys
import sqlite3
import time
import pycountry_convert as pc  # Added for continent mapping
from src.config import LOGS_DIR, GEOCACHE_DB

# Initialize logging
logging.basicConfig(level=logging.INFO, filename=os.path.join(LOGS_DIR, 'app.log'), format='%(asctime)s %(levelname)s:%(message)s')

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Initialize geolocator with increased timeout and user agent
GEOCODER_USER_AGENT = os.getenv('GEOCODER_USER_AGENT', 'location_data_geocoding_tool')
geolocator = Nominatim(user_agent=GEOCODER_USER_AGENT, timeout=10)

# Initialize or connect to the SQLite database for caching
conn = sqlite3.connect(GEOCACHE_DB, check_same_thread=False)
cursor = conn.cursor()

# Create a table for forward geocoding cache if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS geocache (
    location TEXT PRIMARY KEY,
    latitude REAL,
    longitude REAL
)
''')

# Create a table for reverse geocoding cache if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS reverse_geocache (
    lat REAL,
    lon REAL,
    granularity TEXT,
    value TEXT,
    PRIMARY KEY (lat, lon, granularity)
)
''')

conn.commit()


######################################################################



def detect_geographical_columns(df: pd.DataFrame) -> list:
    """Detect columns that likely contain geographical data based on keywords."""
    # Keywords that indicate geographical information
    geo_keywords = ['city', 'country', 'suburb', 'region', 'state', 'province', 'address', 'location', 'place', 'geo', 'zipcode', 'postal', 'district', 'town', 'name']
    # List to store the names of columns that likely contain geographical data
    geo_columns = []
    # Loop through columns in the DataFrame
    for col in df.columns:
        # Skip numerical columns
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        # Check if any keyword appears in the column name (case insensitive)
        for keyword in geo_keywords:
            if keyword.lower() in col.lower():
                geo_columns.append(col)
                break

    return geo_columns

#####################################################################


def extract_gpe_entities(text):
    """Extract GPE entities using spaCy NER."""
    try:
        doc = nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    except Exception as e:
        logging.error(f"Error extracting GPE entities from text '{text}': {e}")
        return []

#####################################################################


def geocode_location(location):
    """Geocode a single location string."""
    try:
        loc = geolocator.geocode(location)
        if loc:
            logging.info(f"Geocoded '{location}': Latitude={loc.latitude}, Longitude={loc.longitude}")
            return {
                'latitude': loc.latitude,
                'longitude': loc.longitude
            }
        else:
            logging.warning(f"Geocoding returned no result for location: '{location}'")
            return {'latitude': None, 'longitude': None}
    except GeocoderTimedOut:
        logging.error(f"Geocoding timed out for location: '{location}'")
        return {'latitude': None, 'longitude': None}
    except Exception as e:
        logging.error(f"Error geocoding location '{location}': {e}")
        return {'latitude': None, 'longitude': None}

#####################################################################

def geocode_location_with_cache(location):
    """Geocode a location with caching using SQLite."""
    try:
        cursor.execute("SELECT latitude, longitude FROM geocache WHERE location = ?", (location,))
        result = cursor.fetchone()

        if result:
            logging.info(f"Cache hit for location '{location}'")
            return {
                'latitude': result[0],
                'longitude': result[1]
            }
        else:
            logging.info(f"Cache miss for location '{location}'. Geocoding...")
            geocoded = geocode_location(location)
            cursor.execute('''
                INSERT INTO geocache (location, latitude, longitude)
                VALUES (?, ?, ?)
            ''', (
                location,
                geocoded['latitude'],
                geocoded['longitude']
            ))
            conn.commit()
            return geocoded
    except Exception as e:
        logging.error(f"Error accessing cache for location '{location}': {e}")
        return {'latitude': None, 'longitude': None}



def perform_geocoding(data: pd.DataFrame, selected_geo_columns: list, session_state, progress_bar, status_text) -> pd.DataFrame:

    """
    Perform geocoding on the selected columns of the DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        selected_geo_columns (list): Columns selected for geocoding.
        session_state: Streamlit session state.
        progress_bar: Streamlit progress bar object.
        status_text: Streamlit status text object.

    Returns:
        pd.DataFrame: DataFrame with added latitude and longitude columns.
    """
    geocoded_df = data.copy()
    # Collect all unique locations across selected columns
    unique_locations = set()
    for column in selected_geo_columns:
        unique_locations.update(geocoded_df[column].dropna().unique())

    unique_locations = list(unique_locations)
    total_locations = len(unique_locations)

    if total_locations == 0:
        raise ValueError("No locations found in the selected columns.")

    # Clear previous geocoded data
    session_state.geocoded_dict = {}

    for idx, loc in enumerate(unique_locations):
        geocoded = interpret_location(loc)
        session_state.geocoded_dict[loc] = geocoded

        # Update progress bar
        progress = (idx + 1) / total_locations
        progress = min(progress, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Geocoding {idx + 1}/{total_locations} locations...")

        # Respect Nominatim's rate limit: 1 request per second
        time.sleep(1)

    # Apply geocoded results to the dataframe
    for column in selected_geo_columns:
        # Add Latitude and Longitude
        geocoded_df[f'Latitude from {column}'] = geocoded_df[column].apply(
            lambda x: session_state.geocoded_dict.get(x, {}).get('latitude', None) if pd.notnull(x) else None
        )
        geocoded_df[f'Longitude from {column}'] = geocoded_df[column].apply(
            lambda x: session_state.geocoded_dict.get(x, {}).get('longitude', None) if pd.notnull(x) else None
        )

    return geocoded_df

#####################################################################

def interpret_location(text):
    """Interpret location using NER and geocoding with cache."""
    try:
        gpe_entities = extract_gpe_entities(text)
        if gpe_entities:
            # Assume the first GPE entity is the relevant location
            location_query = f"{gpe_entities[0]}"
            geocoded = geocode_location_with_cache(location_query)
        else:
            # Fallback to full text geocoding
            location_query = f"{text}"
            geocoded = geocode_location_with_cache(location_query)

        return geocoded  # Return the entire geocoded dict
    except Exception as e:
        logging.error(f"Error interpreting location '{text}': {e}")
        return {'latitude': None, 'longitude': None}


def close_cache_connection():
    """Close the SQLite cache connection."""
    try:
        conn.close()
        logging.info("Cache connection closed.")
    except Exception as e:
        logging.error(f"Error closing cache connection: {e}")

#####################################################################

def reverse_geocode_with_cache(lat, lon, granularity):
    """Reverse geocode latitude and longitude with caching."""
    try:
        # Round the coordinates to 5 decimal places to improve cache hits
        lat_rounded = round(lat, 5)
        lon_rounded = round(lon, 5)

        cursor.execute('''SELECT value FROM reverse_geocache WHERE lat = ? AND lon = ? AND granularity = ?''',
                       (lat_rounded, lon_rounded, granularity))
        result = cursor.fetchone()

        if result:
            logging.info(f"Reverse geocode cache hit for ({lat_rounded}, {lon_rounded}) at granularity '{granularity}'")
            return result[0]
        else:
            # Perform reverse geocoding
            try:
                location = geolocator.reverse((lat_rounded, lon_rounded), exactly_one=True)
                if location and location.raw and 'address' in location.raw:
                    address = location.raw['address']
                    value = None

                    if granularity == 'address':
                        value = location.address
                    elif granularity == 'suburb':
                        value = address.get('suburb') or address.get('neighbourhood') or address.get('hamlet') or address.get('village')
                    elif granularity == 'city':
                        value = address.get('city') or address.get('town') or address.get('municipality')
                    elif granularity == 'state':
                        value = address.get('state') or address.get('region')
                    elif granularity == 'country':
                        value = address.get('country')
                    elif granularity == 'continent':
                        country = address.get('country')
                        if country:
                            try:
                                country_code = pc.country_name_to_country_alpha2(country, cn_name_format="default")
                                continent_code = pc.country_alpha2_to_continent_code(country_code)
                                continent_name = pc.convert_continent_code_to_continent_name(continent_code)
                                value = continent_name
                            except Exception as e:
                                logging.error(f"Error converting country '{country}' to continent: {e}")
                                value = 'Unknown'
                        else:
                            value = 'Unknown'
                    else:
                        value = 'Unknown'  # Assign 'Unknown' for unsupported granularity
                else:
                    value = 'Missing'  # Assign 'Missing' if reverse geocoding fails
            except Exception as e:
                logging.error(f"Error in reverse geocoding for ({lat}, {lon}) with granularity '{granularity}': {e}")
                value = 'Missing'

            # Store in cache
            try:
                cursor.execute('''INSERT OR REPLACE INTO reverse_geocache (lat, lon, granularity, value) VALUES (?, ?, ?, ?)''',
                               (lat_rounded, lon_rounded, granularity, value))
                conn.commit()
                logging.info(f"Reverse geocoded ({lat_rounded}, {lon_rounded}) at granularity '{granularity}': {value}")
            except Exception as e:
                logging.error(f"Error inserting reverse geocode into cache for ({lat_rounded}, {lon_rounded}) at granularity '{granularity}': {e}")

            return value
    except Exception as e:
        logging.error(f"Error accessing reverse geocode cache for ({lat}, {lon}) at granularity '{granularity}': {e}")
        return 'Missing'
    
#####################################################################

def generate_granular_location(data: pd.DataFrame, granularity: str, session_state, progress_bar, status_text, column) -> pd.DataFrame:
    """
    Generate a granular location column based on the specified granularity.

    Args:
        data (pd.DataFrame): The geocoded DataFrame.
        granularity (str): The level of granularity (e.g., address, suburb).
        session_state: Streamlit session state.
        progress_bar: Streamlit progress bar object.
        status_text: Streamlit status text object.
        column (str): The name of the granular location column to be created.

    Returns:
        pd.DataFrame: DataFrame with the new granular location column added.
    """
    
    # Convert the column to string to avoid issues with categorical dtype
    data[column] = data[column].astype(str)

    # Identify latitude and longitude columns
    lat_cols = [col for col in data.columns if col.startswith('Latitude from')]
    lon_cols = [col for col in data.columns if col.startswith('Longitude from')]

    if not lat_cols or not lon_cols:
        raise ValueError("No latitude and longitude columns found.")

    # Use the first pair of latitude and longitude columns
    lat_col = lat_cols[0]
    lon_col = lon_cols[0]

    unique_coords = data[[lat_col, lon_col]].dropna().drop_duplicates()
    total_unique = len(unique_coords)

    if total_unique == 0:
        raise ValueError("No valid (latitude, longitude) pairs found.")

    # Collect granular data
    granular_data = []

    for count, (idx, row) in enumerate(unique_coords.iterrows()):
        lat = row[lat_col]
        lon = row[lon_col]
        value = reverse_geocode_with_cache(lat, lon, granularity)
        if not value:
            value = "Missing"  # Fill missing values
        granular_data.append({lat_col: lat, lon_col: lon, column: value})

        # Update progress
        progress = (count + 1) / total_unique
        progress = min(progress, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Reverse Geocoding {count + 1}/{total_unique} unique coordinate sets...")

        # Respect Nominatim's rate limit: 1 request per second
        time.sleep(1)
    
    # Create a DataFrame from the granular data
    granular_data = pd.DataFrame(granular_data)
    #write to csv
    granular_data.to_csv('granular_data.csv', index=False)
    # if combination of lat and lon for granular data is in data, then replace the respective "column" with the value
    for idx, row in granular_data.iterrows():
        data.loc[(data[lat_col] == row[lat_col]) & (data[lon_col] == row[lon_col]), column] = row[column]

    #turn the column into a category
    data[column] = data[column].astype('category')

    return data

#####################################################################

def prepare_map_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for mapping by extracting all latitude and longitude pairs.

    Args:
        data (pd.DataFrame): The geocoded DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing 'lat' and 'lon' columns for mapping.
    """
    # Dynamically identify all latitude and longitude columns
    latitude_cols = [col for col in data.columns if col.startswith('Latitude from')]
    longitude_cols = [col for col in data.columns if col.startswith('Longitude from')]

    if not latitude_cols or not longitude_cols:
        raise ValueError("Geocoded data does not contain any 'Latitude from <column>' and 'Longitude from <column>' columns.")

    # Prepare a DataFrame to hold all latitude and longitude pairs
    map_data = pd.DataFrame(columns=['lat', 'lon'])

    for lat_col in latitude_cols:
        # Extract the original column name
        original_column = lat_col.split('Latitude from ')[1]
        lon_col = f'Longitude from {original_column}'

        if lon_col not in longitude_cols:
            logging.warning(f"Missing corresponding longitude column for {lat_col}. Skipping this pair.")
            continue

        # Extract data
        temp_df = data[[lat_col, lon_col]].copy()
        temp_df = temp_df.rename(columns={
            lat_col: 'lat',
            lon_col: 'lon'
        })

        temp_df = temp_df[['lat', 'lon']].dropna(subset=['lat', 'lon'])

        map_data = pd.concat([map_data, temp_df], ignore_index=True)

    if map_data.empty:
        raise ValueError("No valid location data available to display on the map.")

    return map_data


# Ensure the cache connection is closed when the program exits
import atexit
atexit.register(close_cache_connection)
