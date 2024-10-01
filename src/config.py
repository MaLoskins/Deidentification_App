# src/config.py

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
OUTPUT_DIR = os.path.join(DATA_DIR, 'outputs')
GEOCACHE_DB = os.path.join(DATA_DIR, 'geocache.db')

# Subdirectories under outputs
PROCESSED_DATA_DIR = os.path.join(OUTPUT_DIR, 'processed_data')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
UNIQUE_IDENTIFICATIONS_DIR = os.path.join(OUTPUT_DIR, 'unique_identifications')
CAT_MAPPING_DIR = os.path.join(OUTPUT_DIR, 'category_mappings')

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(UNIQUE_IDENTIFICATIONS_DIR, exist_ok=True)
os.makedirs(CAT_MAPPING_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

