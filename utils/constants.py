"""
Constants for SafeZone application.
Contains configuration values, paths, and other constants used throughout the application.
"""

# H3 Configuration
H3_RESOLUTION = 6
H3_NEIGHBOR_RING = 1  # Number of rings to consider for neighboring hexes

# File Paths
DATA_DIR = "data"
CRIME_DATA_FILE = f"{DATA_DIR}/crime_data.tsv"
SEVERITY_DATA_FILE = f"{DATA_DIR}/severity_data.json"

# Crime Severity Scoring (1-10 scale to match existing system)
CRIME_SEVERITY_WEIGHTS = {
    # High severity crimes (8-10)
    "Murder": 10,
    "Rape": 10,
    "Kidnapping": 9,
    "Armed Robbery": 9,
    "Terrorism": 10,
    "Human Trafficking": 9,
    "Extortion": 8,

    # Medium-high severity crimes (6-7)
    "Assault": 7,
    "Robbery": 6,
    "Burglary": 6,
    "Drug Trafficking": 7,
    "Fraud": 6,
    "Cybercrime": 6,

    # Medium severity crimes (4-5)
    "Theft": 4,
    "Pickpocketing": 4,
    "Vehicle Theft": 5,
    "Domestic Violence": 5,
    "Stalking": 4,

    # Low severity crimes (1-3)
    "Vandalism": 2,
    "Public Disturbance": 2,
    "Trespassing": 1,
    "Minor Assault": 3,
    "Shoplifting": 2,
    "Noise Violation": 1,
    "Loitering": 1
}

# Time Weight Configuration
TIME_WEIGHT_DECAY_DAYS = 365  # Number of days for time weight decay
TIME_WEIGHT_MIN = 0.1  # Minimum time weight for very old crimes
TIME_WEIGHT_MAX = 1.0  # Maximum time weight for recent crimes

# Victim Count Weight Configuration
VICTIM_WEIGHT_BASE = 1.0  # Base weight for crimes with 0 victims
VICTIM_WEIGHT_MULTIPLIER = 0.2  # Additional weight per victim
VICTIM_WEIGHT_MAX = 3.0  # Maximum victim weight cap

# Severity Level Thresholds (based on 0-100 normalized scale)
SEVERITY_THRESHOLDS = {
    "Very Low": (0, 20),
    "Low": (21, 40),
    "Medium": (41, 60),
    "High": (61, 80),
    "Very High": (81, 100)
}

# Score Normalization
SCORE_MIN = 0.0
SCORE_MAX = 100.0

# Surrounding Area Calculations
SURROUNDING_WEIGHT = 0.15  # Weight factor for surrounding area influence
SURROUNDING_LOG_SCALE = 10  # Logarithmic scaling factor for surrounding adjustments

# API Response Configuration
DEFAULT_SEVERITY_LEVEL = "Very Low"
DEFAULT_CRIME_SCORE = 0.0

# Dataset Generation Configuration (for development/testing)
DATASET_SIZE = 100  # Number of synthetic crime records to generate
DATE_RANGE_MONTHS = 12  # Range in months for synthetic data generation

# Logging Configuration
LOG_SEVERITY_GENERATION = True  # Whether to log severity data generation
LOG_API_REQUESTS = False  # Whether to log API requests (set to True for debugging)

# Cache Configuration
CACHE_ENABLED = True  # Whether to use cached severity data
REGENERATE_CACHE_ON_STARTUP = True  # Whether to regenerate cache on server startup

# Geographic Bounds (Bangladesh approximate bounds)
BANGLADESH_BOUNDS = {
    "min_lat": 20.670883,
    "max_lat": 26.446526,
    "min_lon": 88.084422,
    "max_lon": 92.672721
}

# Error Messages
ERROR_MESSAGES = {
    "CRIME_DATA_NOT_FOUND": "Crime data file not found. Creating empty severity data.",
    "SEVERITY_DATA_NOT_FOUND": "Severity data file not found.",
    "INVALID_COORDINATES": "Invalid coordinates provided.",
    "CACHE_GENERATION_FAILED": "Failed to generate severity data cache."
}

# Success Messages
SUCCESS_MESSAGES = {
    "SEVERITY_DATA_GENERATED": "Generated severity data for {} H3 hexes",
    "CACHE_LOADED": "Severity data cache loaded successfully"
}
