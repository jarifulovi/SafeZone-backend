from datetime import datetime
from utils.constants import (
    CRIME_SEVERITY_WEIGHTS, TIME_WEIGHT_DECAY_DAYS, TIME_WEIGHT_MIN, TIME_WEIGHT_MAX,
    VICTIM_WEIGHT_BASE, VICTIM_WEIGHT_MULTIPLIER, VICTIM_WEIGHT_MAX,
    SEVERITY_THRESHOLDS, SCORE_MAX, SCORE_MIN
)


class CrimeTypeMapper:
    """Maps crime types to severity scores using centralized weights (1-10 scale)"""

    @classmethod
    def get_crime_severity(cls, crime_type: str) -> int:
        """Get severity score for a crime type from constants (1-10 scale)"""
        return CRIME_SEVERITY_WEIGHTS.get(crime_type, 3)  # Default to medium-low if unknown

    @classmethod
    def get_max_severity(cls) -> int:
        """Get maximum possible severity score"""
        return max(CRIME_SEVERITY_WEIGHTS.values())


class TrendCalculator:
    """Calculates crime trends and weights"""

    @staticmethod
    def calculate_time_weight(occurrence_date: str, current_date: datetime = None) -> float:
        """
        Calculate time-based weight for crime incidents.
        Recent crimes get higher weight.
        Uses the original time bucketing approach.
        """
        if current_date is None:
            current_date = datetime.now()

        # Parse the occurrence date
        try:
            crime_date = datetime.fromisoformat(occurrence_date.replace('Z', '+00:00')).replace(tzinfo=None)
        except ValueError:
            # Try parsing without timezone info
            crime_date = datetime.strptime(occurrence_date, '%Y-%m-%d %H:%M:%S')

        # Calculate days difference
        days_diff = (current_date - crime_date).days

        if days_diff < 0:  # Future date, shouldn't happen but handle it
            return 1.0
        elif days_diff <= 7:  # Within a week
            return 1.0
        elif days_diff <= 30:  # Within a month
            return 0.7
        elif days_diff <= 90:  # Within 3 months
            return 0.4
        else:  # Older than 3 months
            return 0.1

    @staticmethod
    def calculate_victim_weight(victim_count: int) -> float:
        """
        Calculate weight based on victim count.
        More victims = higher severity
        Uses the original victim weight approach.
        """
        if victim_count <= 0:
            return 0.5  # Property damage, no direct victims
        elif victim_count == 1:
            return 1.0
        elif victim_count <= 3:
            return 1.5
        elif victim_count <= 5:
            return 2.0
        else:
            return 2.5  # Mass incident

    @staticmethod
    def normalize_score(raw_score: float, max_possible_score: float) -> int:
        """
        Normalize the raw score to 0-100 range
        """
        if max_possible_score <= 0:
            return 0

        normalized = (raw_score / max_possible_score) * 100
        return min(100, max(0, int(normalized)))

    @staticmethod
    def get_severity_level(score: int) -> str:
        """Convert numeric score to severity level using SEVERITY_THRESHOLDS"""
        for level, (min_score, max_score) in SEVERITY_THRESHOLDS.items():
            if min_score <= score <= max_score:
                return level
        return "Very Low"  # Default fallback
