from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class LocationRequest(BaseModel):
    latitude: float
    longitude: float


class SeverityResponse(BaseModel):
    latitude: float
    longitude: float
    h3_hex: str
    crime_score: float
    severity_level: str


class CrimeForecastItem(BaseModel):
    h3_hex: str
    crime_type: str
    probability: float
    expected_time: Optional[datetime] = None
    severity_score: float = 0.0
    confidence: float = 0.0
    future_day: int = 1


class ForecastResponse(BaseModel):
    location: LocationRequest
    forecasts: List[CrimeForecastItem]
    total_predictions: int
    forecast_period_days: int = 7


class CombinedForecastResponse(BaseModel):
    generated_at: datetime
    forecasts: List[CrimeForecastItem]
    total_predictions: int
    forecast_period_days: int = 7


class SeverityHexData(BaseModel):
    h3_hex: str
    crime_score: float


class AllSeverityResponse(BaseModel):
    total_hexes: int
    hexes_with_scores: int
    max_score: float
    min_score: float
    severity_data: List[SeverityHexData]
    generated_at: datetime


class CrimeTypeStats(BaseModel):
    crime_type: str
    count: int
    percentage: float


class TimeStats(BaseModel):
    hour: int
    count: int
    percentage: float


class RegionStats(BaseModel):
    region: str
    count: int
    percentage: float


class CrimeInfoResponse(BaseModel):
    latitude: float
    longitude: float
    h3_hex: str
    analysis_radius_km: float

    # Basic statistics
    total_crimes: int
    total_victims: int
    crimes_analyzed: int

    # Crime type analysis
    top_crime_types: List[CrimeTypeStats]
    most_common_crime: str

    # Time analysis
    peak_crime_hours: List[TimeStats]
    most_dangerous_hour: int
    least_dangerous_hour: int

    # Regional analysis
    affected_regions: List[RegionStats]

    # Victim analysis
    average_victims_per_crime: float
    max_victims_single_incident: int
    crimes_with_victims: int
    crimes_without_victims: int

    # Temporal patterns
    recent_crimes_30_days: int
    recent_crimes_7_days: int
    oldest_crime_date: Optional[datetime]
    newest_crime_date: Optional[datetime]

    # Risk assessment
    crime_density_per_km2: float
    risk_level: str

    generated_at: datetime
