import h3
import pandas as pd
import json
import os
import math
from datetime import datetime
from schema.schemas import SeverityResponse, AllSeverityResponse, SeverityHexData
from utils.constants import H3_RESOLUTION
from utils.crime_utils import CrimeTypeMapper, TrendCalculator


class SeverityService:
    @staticmethod
    def generate_severity_data() -> None:
        """
        Generate severity data for all H3 hexes in the crime dataset.
        Creates a cached file in data/severity_data.json for fast lookups.
        Removes previous file if exists to avoid duplicates.
        """
        # Remove previous severity data file if exists
        severity_file_path = 'data/severity_data.json'
        if os.path.exists(severity_file_path):
            os.remove(severity_file_path)

        # Load crime data
        try:
            crime_df = pd.read_csv('data/crime_data.tsv', sep='\t')
        except FileNotFoundError:
            print("Warning: Crime data file not found. Creating empty severity data.")
            with open(severity_file_path, 'w') as f:
                json.dump({}, f)
            return

        # Get all unique H3 hexes from crime data
        unique_hexes = crime_df['h3_index'].unique()
        severity_data = {}
        current_date = datetime.now()
        all_scores = []  # Track all scores for normalization

        for hex_id in unique_hexes:
            # Get crimes in this hex and neighboring hexes
            target_hexes = [hex_id] + h3.grid_ring(hex_id, 1)
            relevant_crimes = crime_df[crime_df['h3_index'].isin(target_hexes)]

            if relevant_crimes.empty:
                severity_data[hex_id] = {
                    "h3_hex": hex_id,
                    "crime_score": 0.0
                }
                all_scores.append(0.0)
                continue

            # Calculate weighted crime score
            total_score = 0.0

            for _, crime in relevant_crimes.iterrows():
                # Get base severity from crime type
                crime_severity = CrimeTypeMapper.get_crime_severity(crime['crime_type'])

                # Apply time weight (recent crimes matter more)
                time_weight = TrendCalculator.calculate_time_weight(crime['occurrence'], current_date)

                # Apply victim count weight
                victim_weight = TrendCalculator.calculate_victim_weight(crime['victim_count'])

                # Calculate weighted score for this crime
                crime_score = crime_severity * time_weight * victim_weight
                total_score += crime_score

            severity_data[hex_id] = {
                "h3_hex": hex_id,
                "crime_score": float(total_score)
            }
            all_scores.append(total_score)

        # Normalize all scores to 0-100 range based on actual max score
        if all_scores:
            max_actual_score = max(all_scores)
            if max_actual_score > 0:
                for hex_id in severity_data:
                    raw_score = severity_data[hex_id]["crime_score"]
                    normalized_score = min(100.0, (raw_score / max_actual_score) * 100)
                    severity_data[hex_id]["crime_score"] = round(normalized_score, 2)

        # Save severity data to JSON file
        with open(severity_file_path, 'w') as f:
            json.dump(severity_data, f, indent=2)

        print(f"Generated severity data for {len(severity_data)} H3 hexes")

    @staticmethod
    def calculate_severity(lat: float, lon: float) -> dict:
        """
        Quick severity calculation for map view using cached data.
        Returns minimal info needed for coloring hex or marker.
        """
        hex_id = h3.latlng_to_cell(lat, lon, H3_RESOLUTION)

        # Load cached severity data
        try:
            with open('data/severity_data.json', 'r') as f:
                severity_data = json.load(f)
        except FileNotFoundError:
            return {
                "h3_hex": hex_id,
                "crime_score": 0.0
            }

        # Check if we have data for this hex
        if hex_id in severity_data:
            return severity_data[hex_id]

        # Check neighboring hexes if exact hex not found
        neighboring_hexes = h3.grid_ring(hex_id, 1)
        for neighbor_hex in neighboring_hexes:
            if neighbor_hex in severity_data:
                return severity_data[neighbor_hex]

        # No data found
        return {
            "h3_hex": hex_id,
            "crime_score": 0.0
        }

    @staticmethod
    def calculate_stats(lat: float, lon: float) -> SeverityResponse:
        """
        Detailed stats calculation when user requests info.
        Returns full SeverityResponse with surrounding area adjustments.
        """
        hex_id = h3.latlng_to_cell(lat, lon, H3_RESOLUTION)

        # Get base severity data from cache
        base_severity_data = SeverityService.calculate_severity(lat, lon)
        base_score = base_severity_data["crime_score"]

        # Load cached severity data for surrounding adjustments
        try:
            with open('data/severity_data.json', 'r') as f:
                severity_data = json.load(f)
        except FileNotFoundError:
            # If no cache, return base data with calculated severity level
            base_severity_level = TrendCalculator.get_severity_level(int(base_score))
            return SeverityResponse(
                latitude=lat,
                longitude=lon,
                h3_hex=hex_id,
                crime_score=base_score,
                severity_level=base_severity_level
            )

        # Calculate surrounding area adjustments
        surrounding_hexes = h3.grid_ring(hex_id, 1)  # Get immediate neighbors
        surrounding_bonus = 0.0
        surrounding_weight = 0.15  # 15% weight for surrounding areas

        valid_surrounding_count = 0
        for surrounding_hex in surrounding_hexes:
            if surrounding_hex in severity_data:
                surrounding_score = severity_data[surrounding_hex]["crime_score"]
                surrounding_bonus += surrounding_score * surrounding_weight
                valid_surrounding_count += 1

        # Average the surrounding bonus if we have valid neighbors
        if valid_surrounding_count > 0:
            surrounding_bonus = surrounding_bonus / valid_surrounding_count

        # Apply surrounding adjustment (only additive, no negative impact)
        # Use logarithmic scaling to prevent excessive inflation
        adjustment_factor = math.log(1 + surrounding_bonus / 100) * 10  # Gentle logarithmic scaling

        final_score = min(100.0, base_score + adjustment_factor)  # Cap at 100

        # Recalculate severity level based on final score
        final_severity_level = TrendCalculator.get_severity_level(int(final_score))

        return SeverityResponse(
            latitude=lat,
            longitude=lon,
            h3_hex=hex_id,
            crime_score=round(final_score, 2),
            severity_level=final_severity_level
        )

    @staticmethod
    def get_all_severity_data() -> AllSeverityResponse:
        """
        Get all severity data from cache with summary statistics.
        Returns the complete severity dataset for mapping/visualization.
        """
        try:
            with open('data/severity_data.json', 'r') as f:
                severity_data = json.load(f)
        except FileNotFoundError:
            # Return empty response if no cache exists
            return AllSeverityResponse(
                total_hexes=0,
                hexes_with_scores=0,
                max_score=0.0,
                min_score=0.0,
                severity_data=[],
                generated_at=datetime.now()
            )

        # Convert to list of SeverityHexData objects
        hex_data_list = []
        scores = []

        for hex_id, data in severity_data.items():
            hex_data = SeverityHexData(
                h3_hex=data["h3_hex"],
                crime_score=data["crime_score"]
            )
            hex_data_list.append(hex_data)
            scores.append(data["crime_score"])

        # Calculate summary statistics
        total_hexes = len(hex_data_list)
        hexes_with_scores = len([score for score in scores if score > 0])
        max_score = max(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0

        return AllSeverityResponse(
            total_hexes=total_hexes,
            hexes_with_scores=hexes_with_scores,
            max_score=max_score,
            min_score=min_score,
            severity_data=hex_data_list,
            generated_at=datetime.now()
        )
