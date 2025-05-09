
"""
Configuration settings for the ML API and ML pipeline.
"""
import os
import yaml
from pathlib import Path

# Base directory for the application (project root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to single config file
CONFIG_PATH = os.environ.get("ML_CONFIG_PATH", os.path.join(BASE_DIR, "config", "config.yaml"))

# Load the configuration
try:
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = yaml.safe_load(f)
except Exception as e:
    print(f"Error loading configuration: {str(e)}")
    CONFIG = {}

# Extract API-specific configs
API_CONFIG = CONFIG.get('api', {})
ML_CONFIG = CONFIG

# Directory settings from config or defaults
DATA_DIR = os.environ.get("ML_API_DATA_DIR", os.path.join(BASE_DIR, "data"))
MODEL_DIR = os.path.join(DATA_DIR, "training", "output")

# Ensure directories exist
for directory in [DATA_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# Output file requirements
REQUIRED_OUTPUT_FILES = API_CONFIG.get('required_files', [
    {"name": "metadata.json", "min_size": 100},
    {"name": "training.log", "min_size": 10}
])

# Valid model_id pattern (alphanumeric with optional hyphens)
MODEL_ID_PATTERN = API_CONFIG.get('model_id_pattern', r'^[a-zA-Z0-9-_]{3,50}$')

# API version
API_VERSION = API_CONFIG.get('version', "v1")

# Logging configuration
LOGGING_CONFIG = API_CONFIG.get('logging', {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "maxBytes": 10485760, # 10MB
            "backupCount": 5
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
})
