import pandas as pd
import numpy as np
import os
from datetime import datetime
import math

class FireFeaturePipeline:
    def __init__(self, output_dir="data/processed"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def preprocess(self, df):
        """
        Cleans and normalizes the raw enriched data.
        """
        df = df.copy()
        
        # Handle missing values
        # For numeric cols, fill with mean or reasonable defaults
        defaults = {
            'NDVI': 0.0,
            'NBR': 0.0,
            'LST': 300.0,
            'Temperature_C': 25.0,
            'Humidity': 50.0,
            'Wind_Speed_kmh': 10.0,
            'Precipitation_mm': 0.0,
            'Slope': 0.0,
            'Elevation': 0.0,
            'frp': 0.0
        }
        for col, val in defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
                
        # Drop rows where critical data might still be missing
        return df

    def create_temporal_features(self, df):
        """
        Encodes temporal information (Seasonality).
        """
        df['date'] = pd.to_datetime(df['date'])
        
        # Month Encoding (Cyclical)
        df['month'] = df['date'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Day of Year
        df['day_of_year'] = df['date'].dt.dayofyear
        
        return df

    def correlate_features(self, df):
        """
        Creates interaction features that are physically meaningful for fire.
        """
        # 1. Hot and Dry Index (Temperature / (Humidity + 1))
        if 'Temperature_C' in df.columns and 'Humidity' in df.columns:
            df['hot_dry_idx'] = df['Temperature_C'] / (df['Humidity'] + 1.0)
            
        # 2. Wind-Slope Interaction (Wind * Slope)
        if 'Wind_Speed_kmh' in df.columns and 'Slope' in df.columns:
             df['wind_slope_interaction'] = df['Wind_Speed_kmh'] * np.exp(df['Slope'] / 10.0)

        # 3. Vegetation Stress (LST / NDVI)
        if 'LST' in df.columns and 'NDVI' in df.columns:
            # Avoid div by zero
            df['veg_stress'] = df['LST'] / (df['NDVI'].replace(0, 0.01))
            
        return df

    def create_labels(self, df, lookahead_days=0):
        """
        Creates the target label.
        Rule: If FRP > 0 (indicating active fire) -> Label = 1.
        Future: If lookahead_days > 0, this would check future rows.
        """
        # For now, we assume the row represents "Conditions at T=0".
        # If FRP > 0, it's a fire event.
        # If we have synthetic negatives with FRP=0, they become Label=0.
        
        df['label'] = (df['frp'] > 0).astype(int)
        
        # Risk Categories
        conditions = [
            (df['label'] == 0),
            (df['frp'] < 10),
            (df['frp'] >= 10)
        ]
        choices = ['NO_RISK', 'LOW_RISK', 'HIGH_RISK']
        df['risk_category'] = np.select(conditions, choices, default='NO_RISK')
        
        return df

    def augment_negative_samples(self, df):
        """
        Generates synthetic negative samples to balance the dataset.
        Strategy: Copy positive samples, shift date by 6 months (invert season),
        and mock weather/FRP to represent 'No Fire'.
        """
        negatives = df.copy()
        
        # 1. Invert Season (approx 6 month shift)
        negatives['date'] = negatives['date'] + pd.DateOffset(months=6)
        
        # 2. Set Target to 0
        negatives['frp'] = 0.0
        negatives['satellite'] = 'Synthetic (Neg)'
        
        # 3. Mock Environmental factors to be "Safer"
        # Cooler
        if 'Temperature_C' in negatives.columns:
            negatives['Temperature_C'] = negatives['Temperature_C'] - 10.0
        # Wetter
        if 'Humidity' in negatives.columns:
            negatives['Humidity'] = negatives['Humidity'] + 20.0
            negatives['Humidity'] = negatives['Humidity'].clip(upper=100)
        # Less Wind
        if 'Wind_Speed_kmh' in negatives.columns:
            negatives['Wind_Speed_kmh'] = negatives['Wind_Speed_kmh'] * 0.5
        # Lower LST
        if 'LST' in negatives.columns:
            negatives['LST'] = negatives['LST'] - 10.0
            
        return pd.concat([df, negatives], ignore_index=True)

    def run(self, input_path):
        """
        Executes the full pipeline.
        """
        if not os.path.exists(input_path):
            return {"error": "Input file not found"}
            
        print(f"Loading raw data from {input_path}...")
        df = pd.read_csv(input_path)
        
        # 1. Cleaning
        df = self.preprocess(df)
        
        # 2. Data Augmentation (Negatives)
        # Only if we have mostly positives
        if df['frp'].mean() > 0:
            print("Augmenting with synthetic negative samples...")
            df = self.augment_negative_samples(df)
            
        # 3. Feature Engineering
        df = self.create_temporal_features(df)
        df = self.correlate_features(df)
        
        # 4. Labeling
        df = self.create_labels(df)
        
        # 5. Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"fire_features_{timestamp}.csv"
        out_path = os.path.join(self.output_dir, out_name)
        
        # Save ML-ready CSV (drop non-feature cols if needed, but keeping for traceability)
        df.to_csv(out_path, index=False)
        
        print(f"Feature Engineering complete. Saved to {out_path}")
        return {
            "status": "success",
            "file": out_path,
            "rows": len(df),
            "columns": list(df.columns)
        }

if __name__ == "__main__":
    # Test run
    pipeline = FireFeaturePipeline()
    # Find a csv in data/historical
    hist_dir = "data/historical"
    if os.path.exists(hist_dir):
        files = [f for f in os.listdir(hist_dir) if f.endswith(".csv")]
        if files:
            last_file = os.path.join(hist_dir, files[-1])
            pipeline.run(last_file)
