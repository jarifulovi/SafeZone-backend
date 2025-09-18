import pandas as pd
import numpy as np
import os
import h3
import json
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error
from utils.constants import H3_RESOLUTION, CRIME_DATA_FILE, SEVERITY_DATA_FILE


class CrimeForecastModel:
    """
    Crime forecasting model using Random Forest to predict:
    - Crime type probability for each hex
    - Occurrence time patterns for next 7 days
    """

    def __init__(self):
        self.crime_type_model = None
        self.occurrence_time_model = None
        self.probability_model = None
        self.crime_type_encoder = LabelEncoder()
        self.hex_encoder = LabelEncoder()
        self.feature_columns = []
        self.trained = False

    def load_and_prepare_data(self):
        """Load crime data and prepare features for training"""
        print("Loading crime data...")

        # Get absolute paths for data files
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)

        crime_file_path = os.path.join(project_root, CRIME_DATA_FILE)
        severity_file_path = os.path.join(project_root, SEVERITY_DATA_FILE)

        # Check if files exist
        if not os.path.exists(crime_file_path):
            print(f"Error: Crime data file not found at {crime_file_path}")
            return pd.DataFrame()

        if not os.path.exists(severity_file_path):
            print(f"Error: Severity data file not found at {severity_file_path}")
            return pd.DataFrame()

        # Load crime data
        crime_df = pd.read_csv(crime_file_path, sep='\t')
        crime_df['occurrence'] = pd.to_datetime(crime_df['occurrence'])

        # Load severity data for additional features
        with open(severity_file_path, 'r') as f:
            severity_data = json.load(f)

        print(f"Loaded {len(crime_df)} crime records")

        # Create features for each crime record
        features = []
        current_date = datetime.now()

        for _, crime in crime_df.iterrows():
            # Extract temporal features
            crime_date = crime['occurrence']

            # Calculate days from current date (for future prediction simulation)
            days_from_now = (crime_date - current_date).days

            # Use ALL historical data instead of just 7-day window
            # Skip only very old data (more than 365 days old)
            if days_from_now < -365:
                continue

            # Extract time features
            hour = crime_date.hour
            day_of_week = crime_date.weekday()  # 0=Monday, 6=Sunday
            day_of_month = crime_date.day
            month = crime_date.month

            # Get hex features
            hex_id = crime['h3_index']
            severity_score = severity_data.get(hex_id, {}).get('crime_score', 0)

            # Get neighboring hex information
            neighbors = h3.grid_ring(hex_id, 1)
            neighbor_scores = [severity_data.get(n, {}).get('crime_score', 0) for n in neighbors]
            avg_neighbor_score = np.mean(neighbor_scores) if neighbor_scores else 0
            max_neighbor_score = max(neighbor_scores) if neighbor_scores else 0

            # Simulate future day prediction for training
            # Map historical data to future day patterns (1-7)
            future_day = (abs(days_from_now) % 7) + 1  # Cycle through 1-7

            # Create feature vector
            feature_row = {
                'hex_id': hex_id,
                'severity_score': severity_score,
                'avg_neighbor_score': avg_neighbor_score,
                'max_neighbor_score': max_neighbor_score,
                'hour': hour,
                'day_of_week': day_of_week,
                'day_of_month': day_of_month,
                'month': month,
                'victim_count': crime['victim_count'],
                'crime_type': crime['crime_type'],
                'occurrence_hour': hour,
                'future_day': future_day
            }

            features.append(feature_row)

        # Convert to DataFrame
        self.feature_df = pd.DataFrame(features)
        print(f"Prepared {len(self.feature_df)} feature records for training")

        return self.feature_df

    def engineer_features(self, df):
        """Create additional engineered features"""
        # Encode categorical variables
        df_encoded = df.copy()

        # Encode hex_id
        df_encoded['hex_encoded'] = self.hex_encoder.fit_transform(df_encoded['hex_id'])

        # Create time-based features
        df_encoded['is_weekend'] = (df_encoded['day_of_week'] >= 5).astype(int)
        df_encoded['is_night'] = ((df_encoded['hour'] >= 22) | (df_encoded['hour'] <= 5)).astype(int)
        df_encoded['is_evening'] = ((df_encoded['hour'] >= 18) & (df_encoded['hour'] < 22)).astype(int)
        df_encoded['is_morning'] = ((df_encoded['hour'] >= 6) & (df_encoded['hour'] < 12)).astype(int)

        # Risk category based on severity score
        df_encoded['risk_category'] = pd.cut(df_encoded['severity_score'],
                                           bins=[0, 10, 30, 60, 100],
                                           labels=[0, 1, 2, 3]).astype(int)

        # Define feature columns for training
        self.feature_columns = [
            'hex_encoded', 'severity_score', 'avg_neighbor_score', 'max_neighbor_score',
            'hour', 'day_of_week', 'day_of_month', 'month', 'victim_count',
            'future_day', 'is_weekend', 'is_night', 'is_evening', 'is_morning',
            'risk_category'
        ]

        return df_encoded

    def train_models(self):
        """Train Random Forest models for crime prediction"""
        print("Training crime forecast models...")

        # Load and prepare data
        df = self.load_and_prepare_data()

        if len(df) < 50:
            print("Warning: Not enough data for reliable training!")
            return False

        # Engineer features
        df_encoded = self.engineer_features(df)

        # Check class distribution and filter out classes with too few samples
        crime_counts = df_encoded['crime_type'].value_counts()
        print(f"\nCrime type distribution in training data:")
        for i, (crime, count) in enumerate(crime_counts.head(5).items()):
            print(f"  {i+1}. {crime}: {count} ({count/len(df_encoded)*100:.1f}%)")

        # Filter out crime types with fewer than 2 samples (required for stratification)
        min_samples_required = 2
        valid_crimes = crime_counts[crime_counts >= min_samples_required].index

        print(f"\nFiltering crime types: keeping {len(valid_crimes)} out of {len(crime_counts)} types")
        print(f"Removed {len(crime_counts) - len(valid_crimes)} crime types with < {min_samples_required} samples")

        # Filter the dataset to only include valid crime types
        df_filtered = df_encoded[df_encoded['crime_type'].isin(valid_crimes)].copy()

        if len(df_filtered) < 50:
            print("Error: Not enough data after filtering!")
            return False

        # Prepare training data
        X = df_filtered[self.feature_columns]
        y_crime_type = self.crime_type_encoder.fit_transform(df_filtered['crime_type'])
        y_occurrence_hour = df_filtered['occurrence_hour']

        # Check if we can use stratification
        unique_classes, class_counts = np.unique(y_crime_type, return_counts=True)
        min_class_count = min(class_counts)

        if min_class_count >= 2:
            # Use stratified split
            X_train, X_test, y_type_train, y_type_test, y_hour_train, y_hour_test = train_test_split(
                X, y_crime_type, y_occurrence_hour, test_size=0.2, random_state=42, stratify=y_crime_type
            )
            print(f"Using stratified split with {len(unique_classes)} crime classes")
        else:
            # Use regular split without stratification
            X_train, X_test, y_type_train, y_type_test, y_hour_train, y_hour_test = train_test_split(
                X, y_crime_type, y_occurrence_hour, test_size=0.2, random_state=42
            )
            print(f"Using regular split (stratification not possible)")

        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

        # Train crime type prediction model with class balancing
        print("Training crime type prediction model...")
        self.crime_type_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',  # Balance classes automatically
            random_state=42,
            n_jobs=-1
        )
        self.crime_type_model.fit(X_train, y_type_train)

        # Train occurrence time prediction model
        print("Training occurrence time prediction model...")
        self.occurrence_time_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.occurrence_time_model.fit(X_train, y_hour_train)

        # Train probability model with better normalization
        print("Training probability model...")
        self.probability_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )

        # Create more realistic probability targets
        y_probability = []
        for _, row in X_train.iterrows():
            hex_id = row['hex_encoded']
            future_day = row['future_day']
            severity_score = row['severity_score']

            # Base probability on severity score and temporal factors
            base_prob = min(0.8, severity_score / 100) if severity_score > 0 else 0.1

            # Add some randomness and temporal decay
            temporal_decay = 0.9 ** (future_day - 1)  # Decreases with future days
            random_factor = np.random.uniform(0.7, 1.3)  # Add variability

            probability = min(0.95, max(0.05, base_prob * temporal_decay * random_factor))
            y_probability.append(probability)

        self.probability_model.fit(X_train, y_probability)

        # Evaluate models
        print("\\nModel Evaluation:")

        # Crime type prediction accuracy
        type_pred = self.crime_type_model.predict(X_test)
        type_accuracy = np.mean(type_pred == y_type_test)
        print(f"Crime type prediction accuracy: {type_accuracy:.3f}")

        # Check prediction distribution
        pred_counts = pd.Series(type_pred).value_counts()
        print(f"\\nPrediction distribution (test set):")
        for i, (pred_idx, count) in enumerate(pred_counts.head(5).items()):
            crime_name = self.crime_type_encoder.classes_[pred_idx]
            print(f"  {i+1}. {crime_name}: {count} predictions ({count/len(type_pred)*100:.1f}%)")

        # Occurrence time prediction MAE
        hour_pred = self.occurrence_time_model.predict(X_test)
        hour_mae = mean_absolute_error(y_hour_test, hour_pred)
        print(f"\\nOccurrence time MAE: {hour_mae:.2f} hours")

        # Probability model evaluation
        prob_pred = self.probability_model.predict(X_test)
        print(f"Probability range: {prob_pred.min():.3f} - {prob_pred.max():.3f}")

        # Feature importance
        print("\\nTop 10 important features for crime type prediction:")
        feature_importance = self.crime_type_model.feature_importances_
        for i, importance in enumerate(sorted(zip(self.feature_columns, feature_importance),
                                            key=lambda x: x[1], reverse=True)[:10]):
            feature, imp = importance
            print(f"  {i+1}. {feature}: {imp:.3f}")

        self.trained = True
        print("\\nModel training completed successfully!")
        return True

    def save_models(self, model_dir=None):
        """Save trained models to disk"""
        if model_dir is None:
            # Use absolute path for model directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(script_dir, 'models')

        os.makedirs(model_dir, exist_ok=True)

        if not self.trained:
            print("Error: Models must be trained before saving!")
            return False

        print(f"Saving models to {model_dir}...")

        # Save models
        with open(f'{model_dir}/crime_type_model.pkl', 'wb') as f:
            pickle.dump(self.crime_type_model, f)

        with open(f'{model_dir}/occurrence_time_model.pkl', 'wb') as f:
            pickle.dump(self.occurrence_time_model, f)

        with open(f'{model_dir}/probability_model.pkl', 'wb') as f:
            pickle.dump(self.probability_model, f)

        # Save encoders and metadata
        model_metadata = {
            'crime_type_classes': self.crime_type_encoder.classes_.tolist(),
            'hex_classes': self.hex_encoder.classes_.tolist(),
            'feature_columns': self.feature_columns,
            'trained_date': datetime.now().isoformat()
        }

        with open(f'{model_dir}/model_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)

        print("Models saved successfully!")
        return True

    def load_models(self, model_dir=None):
        """Load trained models from disk"""
        if model_dir is None:
            # Use absolute path for model directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(script_dir, 'models')

        if not os.path.exists(os.path.join(model_dir, 'model_metadata.json')):
            print(f"Error: Model files not found in {model_dir}")
            return False

        print(f"Loading models from {model_dir}...")

        # Load models
        with open(f'{model_dir}/crime_type_model.pkl', 'rb') as f:
            self.crime_type_model = pickle.load(f)

        with open(f'{model_dir}/occurrence_time_model.pkl', 'rb') as f:
            self.occurrence_time_model = pickle.load(f)

        with open(f'{model_dir}/probability_model.pkl', 'rb') as f:
            self.probability_model = pickle.load(f)

        # Load metadata
        with open(f'{model_dir}/model_metadata.json', 'r') as f:
            metadata = json.load(f)

        self.crime_type_encoder.classes_ = np.array(metadata['crime_type_classes'])
        self.hex_encoder.classes_ = np.array(metadata['hex_classes'])
        self.feature_columns = metadata['feature_columns']

        self.trained = True
        print(f"Models loaded successfully! (Trained on: {metadata['trained_date']})")
        return True


def train_and_save_forecast_model():
    """Main function to train and save the forecast model"""
    print("=== SAFEZONE CRIME FORECAST MODEL TRAINING ===")
    print()

    model = CrimeForecastModel()

    # Train the models
    success = model.train_models()

    if success:
        # Save the trained models
        model.save_models()
        print("\\n=== TRAINING COMPLETE ===")
        print("Crime forecast model has been trained and saved successfully!")
        print("The model can now predict:")
        print("  - Crime types for future days (1-7)")
        print("  - Occurrence time patterns")
        print("  - Probability scores for each prediction")
        return True
    else:
        print("\\n=== TRAINING FAILED ===")
        print("Unable to train the model due to insufficient data.")
        return False


if __name__ == "__main__":
    train_and_save_forecast_model()
