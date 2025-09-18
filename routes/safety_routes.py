from fastapi import APIRouter
from schema.schemas import LocationRequest, SeverityResponse, ForecastResponse, CrimeForecastItem, CombinedForecastResponse, AllSeverityResponse, CrimeInfoResponse
from services.safety_service import SeverityService
from services.forecast_service import CrimeForecastService
from services.info_service import InfoService
from datetime import datetime

router = APIRouter()
forecast_service = CrimeForecastService()


@router.get("/")
async def root():
    return {"message": "Welcome to SafeZone Bangladesh Prototype"}


@router.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@router.get("/allSeverity/check", response_model=AllSeverityResponse)
async def all_severity_check():
    """
    Get all severity data for all H3 hexes in Bangladesh
    Returns complete severity cache with summary statistics
    """
    return SeverityService.get_all_severity_data()

@router.post("/severity/check", response_model=SeverityResponse)
async def safety_check(location: LocationRequest):
    return SeverityService.calculate_stats(location.latitude, location.longitude)


@router.post("/forecast/check", response_model=ForecastResponse)
async def forecast_check(location: LocationRequest):
    """
    Get crime forecast for a specific location for the next 7 days
    """
    # Get forecasts from the ML-powered forecast service
    forecasts = forecast_service.get_forecast_for_location(
        location.latitude,
        location.longitude,
        days=7
    )

    # Convert forecast objects to response format
    forecast_items = []
    for forecast in forecasts:
        forecast_item = CrimeForecastItem(
            h3_hex=forecast.h3_hex,
            crime_type=forecast.crime_type,
            probability=forecast.probability,
            expected_time=forecast.expected_time,
            severity_score=forecast.severity_score,
            confidence=forecast.confidence,
            future_day=forecast.future_day
        )
        forecast_items.append(forecast_item)

    return ForecastResponse(
        location=location,
        forecasts=forecast_items,
        total_predictions=len(forecast_items),
        forecast_period_days=7
    )


@router.post("/forecastCombine/check", response_model=CombinedForecastResponse)
async def forecast_combine_check():
    """
    Get combined crime forecast for all of Bangladesh for the next 7 days
    No location needed - returns country-wide forecast
    """
    # Get combined forecasts from the ML-powered forecast service
    combined_forecasts = forecast_service.get_bangladesh_combined_forecast()

    # Convert forecast objects to response format
    forecast_items = []
    for forecast in combined_forecasts:
        forecast_item = CrimeForecastItem(
            h3_hex=forecast.h3_hex,
            crime_type=forecast.crime_type,
            probability=forecast.probability,
            expected_time=forecast.expected_time,
            severity_score=forecast.severity_score,
            confidence=forecast.confidence,
            future_day=forecast.future_day
        )
        forecast_items.append(forecast_item)

    return CombinedForecastResponse(
        generated_at=datetime.now(),
        forecasts=forecast_items,
        total_predictions=len(forecast_items),
        forecast_period_days=7
    )


@router.post("/info/check", response_model=CrimeInfoResponse)
async def info_check(location: LocationRequest):
    return InfoService.get_info(location.latitude, location.longitude)
