import h3
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
from typing import List, Tuple
from schema.schemas import (
    CrimeInfoResponse, CrimeTypeStats, TimeStats, RegionStats
)
from utils.constants import H3_RESOLUTION

class InfoService:

    # Limit for performance - analyze max 5000 most relevant crimes
    MAX_CRIMES_TO_ANALYZE = 5000
    ANALYSIS_RADIUS_KM = 5.0  # 5km radius for analysis

    @staticmethod
    def get_info(lat: float, lon: float) -> CrimeInfoResponse:
        """
        Comprehensive crime analysis for a given location.
        Analyzes crime patterns, trends, and statistics within the area.
        """
        target_hex = h3.latlng_to_cell(lat, lon, H3_RESOLUTION)

        # Load crime data
        try:
            crime_df = pd.read_csv('data/crime_data.tsv', sep='\t')
        except FileNotFoundError:
            return InfoService._create_empty_response(lat, lon, target_hex)

        # Get relevant crimes (target hex + surrounding area)
        relevant_hexes = InfoService._get_analysis_hexes(target_hex, radius_rings=2)
        relevant_crimes = crime_df[crime_df['h3_index'].isin(relevant_hexes)]

        # Limit data for performance
        if len(relevant_crimes) > InfoService.MAX_CRIMES_TO_ANALYZE:
            # Sort by date and take most recent crimes
            relevant_crimes['occurrence'] = pd.to_datetime(relevant_crimes['occurrence'])
            relevant_crimes = relevant_crimes.sort_values('occurrence', ascending=False).head(InfoService.MAX_CRIMES_TO_ANALYZE)
        else:
            relevant_crimes['occurrence'] = pd.to_datetime(relevant_crimes['occurrence'])

        if relevant_crimes.empty:
            return InfoService._create_empty_response(lat, lon, target_hex)

        # Perform comprehensive analysis
        return InfoService._analyze_crime_data(lat, lon, target_hex, relevant_crimes)

    @staticmethod
    def _get_analysis_hexes(center_hex: str, radius_rings: int = 2) -> List[str]:
        """Get hexes within specified radius for analysis"""
        hexes = [center_hex]
        for ring in range(1, radius_rings + 1):
            hexes.extend(h3.grid_ring(center_hex, ring))
        return hexes

    @staticmethod
    def _analyze_crime_data(lat: float, lon: float, hex_id: str, crimes_df: pd.DataFrame) -> CrimeInfoResponse:
        """Perform comprehensive crime data analysis"""
        current_time = datetime.now()
        total_crimes = len(crimes_df)

        # Basic statistics
        total_victims = crimes_df['victim_count'].sum()
        crimes_analyzed = min(total_crimes, InfoService.MAX_CRIMES_TO_ANALYZE)

        # Crime type analysis
        crime_type_counts = crimes_df['crime_type'].value_counts()
        top_crime_types = []
        for crime_type, count in crime_type_counts.head(10).items():
            percentage = (count / total_crimes) * 100
            top_crime_types.append(CrimeTypeStats(
                crime_type=crime_type,
                count=int(count),
                percentage=round(percentage, 2)
            ))

        most_common_crime = crime_type_counts.index[0] if not crime_type_counts.empty else "Unknown"

        # Time analysis (extract hour from occurrence)
        crimes_df['hour'] = crimes_df['occurrence'].dt.hour
        hour_counts = crimes_df['hour'].value_counts().sort_index()

        peak_crime_hours = []
        for hour, count in hour_counts.head(10).items():
            percentage = (count / total_crimes) * 100
            peak_crime_hours.append(TimeStats(
                hour=int(hour),
                count=int(count),
                percentage=round(percentage, 2)
            ))

        most_dangerous_hour = int(hour_counts.idxmax()) if not hour_counts.empty else 0
        least_dangerous_hour = int(hour_counts.idxmin()) if not hour_counts.empty else 0

        # Regional analysis
        region_counts = crimes_df['region_cluster'].value_counts()
        affected_regions = []
        for region, count in region_counts.head(10).items():
            percentage = (count / total_crimes) * 100
            affected_regions.append(RegionStats(
                region=region,
                count=int(count),
                percentage=round(percentage, 2)
            ))

        # Victim analysis
        crimes_with_victims = int((crimes_df['victim_count'] > 0).sum())
        crimes_without_victims = total_crimes - crimes_with_victims
        max_victims = int(crimes_df['victim_count'].max())
        avg_victims = float(crimes_df['victim_count'].mean())

        # Temporal patterns
        recent_30_days = current_time - timedelta(days=30)
        recent_7_days = current_time - timedelta(days=7)

        recent_crimes_30 = int((crimes_df['occurrence'] >= recent_30_days).sum())
        recent_crimes_7 = int((crimes_df['occurrence'] >= recent_7_days).sum())

        oldest_date = crimes_df['occurrence'].min()
        newest_date = crimes_df['occurrence'].max()

        # Risk assessment
        area_km2 = InfoService._calculate_analysis_area()
        crime_density = total_crimes / area_km2 if area_km2 > 0 else 0
        risk_level = InfoService._calculate_risk_level(crime_density, recent_crimes_7)

        return CrimeInfoResponse(
            latitude=lat,
            longitude=lon,
            h3_hex=hex_id,
            analysis_radius_km=InfoService.ANALYSIS_RADIUS_KM,
            total_crimes=total_crimes,
            total_victims=total_victims,
            crimes_analyzed=crimes_analyzed,
            top_crime_types=top_crime_types,
            most_common_crime=most_common_crime,
            peak_crime_hours=peak_crime_hours,
            most_dangerous_hour=most_dangerous_hour,
            least_dangerous_hour=least_dangerous_hour,
            affected_regions=affected_regions,
            average_victims_per_crime=round(avg_victims, 2),
            max_victims_single_incident=max_victims,
            crimes_with_victims=crimes_with_victims,
            crimes_without_victims=crimes_without_victims,
            recent_crimes_30_days=recent_crimes_30,
            recent_crimes_7_days=recent_crimes_7,
            oldest_crime_date=oldest_date,
            newest_crime_date=newest_date,
            crime_density_per_km2=round(crime_density, 2),
            risk_level=risk_level,
            generated_at=current_time
        )

    @staticmethod
    def _calculate_analysis_area() -> float:
        """Calculate approximate analysis area in km²"""
        # Rough approximation: 2 rings around center hex at resolution 6
        # Each hex at resolution 6 is approximately 36.5 km²
        # Center + 1 ring (6 hexes) + 2 ring (12 hexes) = 19 hexes
        return 19 * 36.5

    @staticmethod
    def _calculate_risk_level(density: float, recent_crimes: int) -> str:
        """Calculate risk level based on crime density and recent activity"""
        if density >= 10 or recent_crimes >= 20:
            return "Very High"
        elif density >= 5 or recent_crimes >= 10:
            return "High"
        elif density >= 2 or recent_crimes >= 5:
            return "Moderate"
        elif density >= 0.5 or recent_crimes >= 2:
            return "Low"
        else:
            return "Very Low"

    @staticmethod
    def _create_empty_response(lat: float, lon: float, hex_id: str) -> CrimeInfoResponse:
        """Create empty response when no crime data is available"""
        return CrimeInfoResponse(
            latitude=lat,
            longitude=lon,
            h3_hex=hex_id,
            analysis_radius_km=InfoService.ANALYSIS_RADIUS_KM,
            total_crimes=0,
            total_victims=0,
            crimes_analyzed=0,
            top_crime_types=[],
            most_common_crime="No Data",
            peak_crime_hours=[],
            most_dangerous_hour=0,
            least_dangerous_hour=0,
            affected_regions=[],
            average_victims_per_crime=0.0,
            max_victims_single_incident=0,
            crimes_with_victims=0,
            crimes_without_victims=0,
            recent_crimes_30_days=0,
            recent_crimes_7_days=0,
            oldest_crime_date=None,
            newest_crime_date=None,
            crime_density_per_km2=0.0,
            risk_level="No Data",
            generated_at=datetime.now()
        )
