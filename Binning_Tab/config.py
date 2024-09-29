# config.py

import os

# Define the base data directory
BASE_DATA_DIR = 'data_storage'

# Define subdirectories
PROCESSED_DATA_DIR = os.path.join(BASE_DATA_DIR, 'processed_data')
REPORTS_DIR = os.path.join(BASE_DATA_DIR, 'reports')
MAPPINGS_DIR = os.path.join(BASE_DATA_DIR, 'mappings')
PLOTS_DIR = os.path.join(BASE_DATA_DIR, 'plots')
LOGS_DIR = os.path.join(BASE_DATA_DIR, 'logs')
