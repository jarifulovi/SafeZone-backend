import json
import pandas as pd
import h3
import numpy as np
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

from utils.constants import H3_RESOLUTION, SEVERITY_DATA_FILE, CRIME_DATA_FILE


@dataclass
class CrimeForecast:
    h3_hex: str
    crime_type: str
    probability: float
    expected_time: Optional[datetime] = None
    severity_score: float = 0.0
    confidence: float = 0.0
    future_day: int = 1


class CrimeForecastService:
    def __init__(self):
        self.crime_df = None
        self.severity_data = None
        self.forecast_model = None

        # Get absolute path for cache file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        self.combined_cache_file = os.path.join(project_root, 'data', 'forecast_combine_cache.json')

        self.load_data()
        self.load_forecast_model()

    def load_data(self):
        """Load crime and severity data using absolute paths"""
        try:
            # Get absolute paths for data files
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)

            crime_file_path = os.path.join(project_root, CRIME_DATA_FILE)
            severity_file_path = os.path.join(project_root, SEVERITY_DATA_FILE)

            print(f"Loading crime data from: {crime_file_path}")
            print(f"Loading severity data from: {severity_file_path}")

            # Check if files exist
            if not os.path.exists(crime_file_path):
                print(f"Error: Crime data file not found at {crime_file_path}")
                self.crime_df = pd.DataFrame()
            else:
                self.crime_df = pd.read_csv(crime_file_path, sep='\t')
                self.crime_df['occurrence'] = pd.to_datetime(self.crime_df['occurrence'])
                print(f"✅ Loaded {len(self.crime_df)} crime records")

            if not os.path.exists(severity_file_path):
                print(f"Error: Severity data file not found at {severity_file_path}")
                self.severity_data = {}
            else:
                with open(severity_file_path, 'r') as f:
                    self.severity_data = json.load(f)
                print(f"✅ Loaded {len(self.severity_data)} severity data entries")

        except Exception as e:
            print(f"Error loading data: {e}")
            self.crime_df = pd.DataFrame()
            self.severity_data = {}

    def load_forecast_model(self):
        """Load the trained ML forecast model"""
        try:
            from data.forecast_model import CrimeForecastModel
            self.forecast_model = CrimeForecastModel()

            if self.forecast_model.load_models():
                print("✅ Crime forecast model loaded successfully!")
            else:
                print("⚠️ Could not load forecast model. Training new model...")
                self.forecast_model.train_models()
                self.forecast_model.save_models()

        except Exception as e:
            print(f"Error loading forecast model: {e}")
            self.forecast_model = None

    def generate_7_day_forecast(self, top_hexes: int = 50) -> List[CrimeForecast]:
        """
        Generate ML-based crime forecasts for the next 7 days
        Returns list of predicted crimes with probabilities
        """
        if not self.forecast_model or not self.forecast_model.trained:
            print("Forecast model not available, using fallback method")
            return self._fallback_forecast()

        forecasts = []
        current_date = datetime.now()

        # Get top crime-prone hexes from severity data
        top_hex_ids = self.get_top_crime_hexes(top_hexes)

        if not top_hex_ids:
            print("⚠️ No high-risk hexes found in severity data")
            return []

        print(f"Generating forecasts for {len(top_hex_ids)} high-risk hexes...")

        for hex_id in top_hex_ids:
            hex_forecasts = self._predict_for_hex_ml(hex_id, current_date)
            forecasts.extend(hex_forecasts)

        # Sort by probability (most likely first)
        forecasts.sort(key=lambda x: x.probability, reverse=True)

        print(f"Generated {len(forecasts)} forecast predictions")
        return forecasts[:100]  # Return top 100 predictions

    def _predict_for_hex_ml(self, hex_id: str, current_date: datetime) -> List[CrimeForecast]:
        """Use ML model to predict crimes for a specific hex"""
        forecasts = []

        # Get hex features
        severity_score = self.severity_data.get(hex_id, {}).get('crime_score', 0)

        # Get neighboring hex information
        neighbors = h3.grid_ring(hex_id, 1)
        neighbor_scores = [self.severity_data.get(n, {}).get('crime_score', 0) for n in neighbors]
        avg_neighbor_score = np.mean(neighbor_scores) if neighbor_scores else 0
        max_neighbor_score = max(neighbor_scores) if neighbor_scores else 0

        # Check if hex is in the model's training data
        if hex_id not in self.forecast_model.hex_encoder.classes_:
            # For unseen hexes, use a similar hex or default prediction
            return self._predict_for_unseen_hex(hex_id, severity_score, current_date)

        # Encode hex_id
        try:
            hex_encoded = self.forecast_model.hex_encoder.transform([hex_id])[0]
        except ValueError:
            return []

        # Predict for each of the next 7 days
        for future_day in range(1, 8):
            target_date = current_date + timedelta(days=future_day)

            # Create feature vector for prediction
            hour_predictions = self._predict_hours_for_day(
                hex_encoded, severity_score, avg_neighbor_score, max_neighbor_score,
                target_date, future_day
            )

            for hour_pred in hour_predictions:
                if hour_pred['probability'] > 0.1:  # Only significant probabilities
                    forecast = CrimeForecast(
                        h3_hex=hex_id,
                        crime_type=hour_pred['crime_type'],
                        probability=round(hour_pred['probability'], 3),
                        expected_time=target_date.replace(
                            hour=int(hour_pred['hour']),
                            minute=np.random.randint(0, 60),
                            second=0,
                            microsecond=0
                        ),
                        severity_score=severity_score,
                        confidence=hour_pred['confidence'],
                        future_day=future_day
                    )
                    forecasts.append(forecast)

        return forecasts

    def _predict_hours_for_day(self, hex_encoded, severity_score, avg_neighbor_score,
                              max_neighbor_score, target_date, future_day):
        """Predict crime occurrences for different hours of a specific day"""
        predictions = []

        # Predict for key time periods (morning, afternoon, evening, night)
        key_hours = [8, 14, 19, 23]  # Representative hours for each period

        for hour in key_hours:
            # Create feature vector
            feature_vector = self._create_feature_vector(
                hex_encoded, severity_score, avg_neighbor_score, max_neighbor_score,
                hour, target_date, future_day
            )

            try:
                # Convert to DataFrame with proper column names for sklearn
                feature_df = pd.DataFrame([feature_vector], columns=self.forecast_model.feature_columns)

                # Predict crime type probabilities for ALL classes
                crime_type_probs = self.forecast_model.crime_type_model.predict_proba(feature_df)[0]

                # Get top 3 most likely crime types instead of just the highest
                top_indices = np.argsort(crime_type_probs)[-3:][::-1]  # Top 3 in descending order

                for i, crime_type_idx in enumerate(top_indices):
                    crime_type = self.forecast_model.crime_type_encoder.classes_[crime_type_idx]
                    type_probability = crime_type_probs[crime_type_idx]

                    # Skip very low probability predictions
                    if type_probability < 0.05:
                        continue

                    # Predict overall probability
                    base_probability = self.forecast_model.probability_model.predict(feature_df)[0]

                    # Combine type probability with base probability
                    combined_probability = base_probability * type_probability

                    # Add temporal and severity adjustments
                    temporal_decay = 0.95 ** (future_day - 1)  # Slight decay for future days
                    severity_boost = min(1.5, 1 + (severity_score / 200))  # Boost for high severity areas

                    # Apply adjustments
                    final_probability = combined_probability * temporal_decay * severity_boost

                    # Clamp between reasonable bounds
                    final_probability = max(0.01, min(0.85, final_probability))

                    # Add some randomness to avoid identical predictions
                    random_factor = np.random.uniform(0.85, 1.15)
                    final_probability *= random_factor
                    final_probability = max(0.01, min(0.85, final_probability))

                    # Predict occurrence hour (fine-tune the hour)
                    predicted_hour = self.forecast_model.occurrence_time_model.predict(feature_df)[0]
                    predicted_hour = max(0, min(23, int(predicted_hour)))

                    # Calculate confidence based on type probability
                    confidence = min(0.9, max(0.1, type_probability * 0.7 + 0.3))

                    predictions.append({
                        'crime_type': crime_type,
                        'probability': final_probability,
                        'hour': predicted_hour,
                        'confidence': confidence
                    })

            except Exception as e:
                print(f"Prediction error for hex {hex_encoded}: {e}")
                continue

        # Sort by probability and return diverse predictions
        predictions.sort(key=lambda x: x['probability'], reverse=True)

        # Ensure diversity - limit same crime type predictions
        diverse_predictions = []
        crime_type_counts = {}

        for pred in predictions:
            crime_type = pred['crime_type']
            if crime_type_counts.get(crime_type, 0) < 2:  # Max 2 of same type per day
                diverse_predictions.append(pred)
                crime_type_counts[crime_type] = crime_type_counts.get(crime_type, 0) + 1

        return diverse_predictions[:6]  # Return top 6 diverse predictions

    def _create_feature_vector(self, hex_encoded, severity_score, avg_neighbor_score,
                              max_neighbor_score, hour, target_date, future_day):
        """Create feature vector for ML model prediction"""
        day_of_week = target_date.weekday()
        day_of_month = target_date.day
        month = target_date.month

        # Calculate engineered features
        is_weekend = 1 if day_of_week >= 5 else 0
        is_night = 1 if (hour >= 22 or hour <= 5) else 0
        is_evening = 1 if (18 <= hour < 22) else 0
        is_morning = 1 if (6 <= hour < 12) else 0

        # Risk category based on severity score
        if severity_score <= 10:
            risk_category = 0
        elif severity_score <= 30:
            risk_category = 1
        elif severity_score <= 60:
            risk_category = 2
        else:
            risk_category = 3

        # Create feature vector matching training data
        feature_vector = [
            hex_encoded,                # hex_encoded
            severity_score,             # severity_score
            avg_neighbor_score,         # avg_neighbor_score
            max_neighbor_score,         # max_neighbor_score
            hour,                       # hour
            day_of_week,               # day_of_week
            day_of_month,              # day_of_month
            month,                     # month
            1,                         # victim_count (default)
            future_day,                # future_day
            is_weekend,                # is_weekend
            is_night,                  # is_night
            is_evening,                # is_evening
            is_morning,                # is_morning
            risk_category              # risk_category
        ]

        return feature_vector

    def _predict_for_unseen_hex(self, hex_id: str, severity_score: float, current_date: datetime) -> List[CrimeForecast]:
        """Handle prediction for hexes not seen during training"""
        forecasts = []

        # Use severity-based simple prediction for unseen hexes
        if severity_score > 0:
            # Simple probability based on severity score
            base_prob = min(0.8, severity_score / 100)

            for future_day in range(1, 4):  # Only predict for next 3 days for unseen hexes
                target_date = current_date + timedelta(days=future_day)

                # Predict common crime types with decreasing probability
                common_crimes = ['Theft', 'Assault', 'Burglary', 'Vandalism']

                for i, crime_type in enumerate(common_crimes):
                    probability = base_prob * (0.8 ** i)  # Decreasing probability

                    if probability > 0.1:
                        forecast = CrimeForecast(
                            h3_hex=hex_id,
                            crime_type=crime_type,
                            probability=round(probability, 3),
                            expected_time=target_date.replace(
                                hour=np.random.randint(8, 22),
                                minute=np.random.randint(0, 60),
                                second=0,
                                microsecond=0
                            ),
                            severity_score=severity_score,
                            confidence=0.5,  # Lower confidence for unseen hexes
                            future_day=future_day
                        )
                        forecasts.append(forecast)

        return forecasts

    def get_top_crime_hexes(self, top_n: int = 50) -> List[str]:
        """Get top N hexes with highest crime scores"""
        if not self.severity_data:
            print("⚠️ No severity data available")
            return []

        sorted_hexes = sorted(
            self.severity_data.items(),
            key=lambda x: x[1]['crime_score'],
            reverse=True
        )

        # Filter out hexes with zero scores
        filtered_hexes = [hex_id for hex_id, data in sorted_hexes if data['crime_score'] > 0]

        print(f"Found {len(filtered_hexes)} hexes with crime scores > 0")
        return filtered_hexes[:top_n]

    def get_forecast_for_location(self, lat: float, lon: float, days: int = 7) -> List[CrimeForecast]:
        """Get ML-based forecast for a specific location"""
        hex_id = h3.latlng_to_cell(lat, lon, H3_RESOLUTION)

        if not self.forecast_model or not self.forecast_model.trained:
            print("⚠️ Forecast model not available")
            return []

        # Generate forecasts for this specific hex
        hex_forecasts = self._predict_for_hex_ml(hex_id, datetime.now())

        # Filter for requested days
        end_date = datetime.now() + timedelta(days=days)
        filtered_forecasts = [
            f for f in hex_forecasts
            if f.expected_time and f.expected_time <= end_date
        ]

        # Sort by probability
        filtered_forecasts.sort(key=lambda x: x.probability, reverse=True)

        return filtered_forecasts

    def _fallback_forecast(self) -> List[CrimeForecast]:
        """Fallback method when ML model is not available"""
        print("Using statistical fallback forecast method...")

        if not self.severity_data:
            return []

        forecasts = []
        current_date = datetime.now()

        # Get top hexes with simple statistical approach
        top_hexes = self.get_top_crime_hexes(20)

        for hex_id in top_hexes:
            severity_score = self.severity_data.get(hex_id, {}).get('crime_score', 0)

            if severity_score > 10:  # Only predict for areas with significant crime
                base_prob = min(0.6, severity_score / 100)

                for future_day in range(1, 4):
                    target_date = current_date + timedelta(days=future_day)

                    # Simple statistical prediction
                    common_crimes = ['Theft', 'Assault', 'Burglary']

                    for i, crime_type in enumerate(common_crimes):
                        probability = base_prob * (0.7 ** i)

                        if probability > 0.1:
                            forecast = CrimeForecast(
                                h3_hex=hex_id,
                                crime_type=crime_type,
                                probability=round(probability, 3),
                                expected_time=target_date.replace(
                                    hour=np.random.randint(10, 20),
                                    minute=0,
                                    second=0,
                                    microsecond=0
                                ),
                                severity_score=severity_score,
                                confidence=0.3,  # Lower confidence for fallback
                                future_day=future_day
                            )
                            forecasts.append(forecast)

        forecasts.sort(key=lambda x: x.probability, reverse=True)
        return forecasts[:50]

    def generate_and_cache_combined_forecast(self) -> bool:
        """
        Generate combined forecast for all of Bangladesh and cache it
        Returns True if successful, False otherwise
        """
        try:
            print("Generating combined Bangladesh forecast...")

            # Generate 7-day forecast for top crime areas
            combined_forecasts = self.generate_7_day_forecast(top_hexes=50)  # Get more areas for BD-wide coverage

            if not combined_forecasts:
                print("No forecasts generated")
                return False

            # Convert forecasts to serializable format
            cache_data = {
                'generated_at': datetime.now().isoformat(),
                'total_predictions': len(combined_forecasts),
                'forecast_period_days': 7,
                'forecasts': []
            }

            for forecast in combined_forecasts:
                forecast_dict = {
                    'h3_hex': forecast.h3_hex,
                    'crime_type': forecast.crime_type,
                    'probability': forecast.probability,
                    'expected_time': forecast.expected_time.isoformat() if forecast.expected_time else None,
                    'severity_score': forecast.severity_score,
                    'confidence': forecast.confidence,
                    'future_day': forecast.future_day
                }
                cache_data['forecasts'].append(forecast_dict)

            # Save to cache file
            with open(self.combined_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

            print(f"✅ Cached {len(combined_forecasts)} combined forecasts to {self.combined_cache_file}")
            return True

        except Exception as e:
            print(f"Error generating combined forecast cache: {e}")
            return False

    def get_combined_forecast_from_cache(self) -> List[CrimeForecast]:
        """
        Retrieve combined forecast from cache
        Returns list of forecasts or empty list if cache is invalid/missing
        """
        try:
            if not os.path.exists(self.combined_cache_file):
                print("Cache file not found, generating new combined forecast...")
                if self.generate_and_cache_combined_forecast():
                    return self.get_combined_forecast_from_cache()  # Retry after generation
                else:
                    return []


            # Load cached data
            with open(self.combined_cache_file, 'r') as f:
                cache_data = json.load(f)

            # Convert back to CrimeForecast objects
            forecasts = []
            for forecast_dict in cache_data.get('forecasts', []):
                expected_time = None
                if forecast_dict.get('expected_time'):
                    expected_time = datetime.fromisoformat(forecast_dict['expected_time'])

                forecast = CrimeForecast(
                    h3_hex=forecast_dict['h3_hex'],
                    crime_type=forecast_dict['crime_type'],
                    probability=forecast_dict['probability'],
                    expected_time=expected_time,
                    severity_score=forecast_dict['severity_score'],
                    confidence=forecast_dict['confidence'],
                    future_day=forecast_dict['future_day']
                )
                forecasts.append(forecast)

            print(f"✅ Loaded {len(forecasts)} forecasts from cache")
            return forecasts

        except Exception as e:
            print(f"Error loading combined forecast cache: {e}")
            # Try to regenerate cache as fallback
            if self.generate_and_cache_combined_forecast():
                return self.get_combined_forecast_from_cache()
            return []

    def get_bangladesh_combined_forecast(self) -> List[CrimeForecast]:
        """
        Get combined forecast for all of Bangladesh
        This is the main method to be called by the API
        """
        return self.get_combined_forecast_from_cache()


# Example usage
if __name__ == "__main__":
    forecast_service = CrimeForecastService()

    # Generate 7-day forecast using ML model
    # forecasts = forecast_service.generate_7_day_forecast()
    #
    # print("\\n=== 7-Day ML Crime Forecast ===")
    # print("=" * 80)
    # for i, forecast in enumerate(forecasts[:]):  # Show top 15
    #     print(f"{i + 1:2d}. {forecast.crime_type:15} | "
    #           f"Location: {forecast.h3_hex} |"
    #           f"Prob: {forecast.probability:.3f} | "
    #           f"Day {forecast.future_day} | "
    #           f"Severity: {forecast.severity_score} | "
    #           f"Time: {forecast.expected_time.strftime('%Y-%m-%d %H:%M')} | "
    #           f"Conf: {forecast.confidence:.2f}")
    #
    # # Test specific location (Dhaka Center)
    # print("\\n=== Forecast for Dhaka Center ===")
    # dhaka_forecasts = forecast_service.get_forecast_for_location(23.8103, 90.4125)
    # for i, forecast in enumerate(dhaka_forecasts[:10]):
    #     print(f"{i + 1}. {forecast.crime_type}: {forecast.probability:.3f} chance on day {forecast.future_day}")

    # Generate and cache combined forecast for Bangladesh
    print("\\n=== Generate Combined Forecast for Bangladesh ===")
    if forecast_service.generate_and_cache_combined_forecast():
        print("Combined forecast generated and cached successfully")
    else:
        print("Failed to generate combined forecast")
