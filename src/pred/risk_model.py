import numpy as np
import math
from datetime import datetime
from src.pred.prediction_model import FirePredictionTrainer

class FireRiskPredictor:
    def __init__(self):
        self.ml_model = FirePredictionTrainer()

    def calculate_risk_score(self, lst_k, ndvi, humidity, wind_speed, nbr=0, swir=0, precip=0, slope=0, elevation=0):
        """
        Calculates fire risk probability (0-1) using Multi-Source Live Data.
        """
        probability = None
        
        # 1. Try ML Model
        try:
            # Construct feature vector matching training schema
            temp_c = lst_k - 273.15
            now = datetime.now()
            day_of_year = now.timetuple().tm_yday
            month = now.month
            
            # Derived Features
            hot_dry_idx = temp_c / (humidity + 1.0)
            wind_slope = wind_speed * np.exp(slope / 10.0)
            veg_stress = lst_k / (ndvi if ndvi > 0 else 0.01)
            
            input_features = {
                'LST': lst_k,
                'NDVI': ndvi,
                'NBR': nbr,
                'SWIR': swir,
                'Temperature_C': temp_c,
                'Humidity': humidity,
                'Wind_Speed_kmh': wind_speed,
                'Precipitation_mm': precip,
                'Slope': slope,
                'Elevation': elevation,
                'day_of_year': day_of_year,
                'month_sin': np.sin(2 * np.pi * month / 12),
                'month_cos': np.cos(2 * np.pi * month / 12),
                'hot_dry_idx': hot_dry_idx,
                'veg_stress': veg_stress,
                'wind_slope_interaction': wind_slope
            }
            
            probability = self.ml_model.predict_risk(input_features)
            
        except Exception as e:
            print(f"ML Inference Error (using heuristic): {e}")

        # 2. Heuristic Fallback
        if probability is None:
            # Heuristic Logic
            temp_c = lst_k - 273.15
            temp_risk = min(max((temp_c - 20) / 25, 0), 1)
            
            # NDVI: Dryness
            if ndvi < 0: ndvi_risk = 0 
            else: ndvi_risk = 1.0 - min(max(ndvi, 0), 1)

            humidity_risk = min(max((60 - humidity) / 40, 0), 1)
            wind_risk = min(wind_speed / 50, 1)
            
            # Slope Factor: Slopes increase spread rate / risk
            slope_risk = min(slope / 30, 0.5)

            probability = (
                (temp_risk * 0.3) + 
                (ndvi_risk * 0.3) + 
                (humidity_risk * 0.2) + 
                (wind_risk * 0.1) +
                (slope_risk * 0.1)
            )
            # Reduce probability if raining
            if precip > 1.0:
                probability *= 0.1
                
            probability = min(max(probability, 0), 1)

        # Common Output Formatting
        level = "LOW"
        if probability > 0.6:
            level = "HIGH"
        elif probability > 0.3:
            level = "MEDIUM"

        return {
            "probability": round(probability, 4),
            "risk_level": level,
            "components": {
                "temp_c": round(lst_k - 273.15, 1),
                "ndvi_val": round(ndvi, 2),
                "humidity_val": humidity,
                "wind_kph": wind_speed,
                "source": "ML_RandomForest" if self.ml_model and probability is not None else "Heuristic"
            }
        }
