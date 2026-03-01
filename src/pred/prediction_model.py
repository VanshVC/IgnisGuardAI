import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from datetime import datetime

class FirePredictionTrainer:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "fire_risk_model.pkl")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.features = [
            'NDVI', 'NBR', 'LST', 'SWIR', 
            'Temperature_C', 'Humidity', 'Wind_Speed_kmh', 'Precipitation_mm', 
            'Slope', 'Elevation',
            'month_sin', 'month_cos', 'day_of_year',
            'hot_dry_idx', 'wind_slope_interaction', 'veg_stress'
        ]

    def train(self, data_path):
        """
        Trains the Random Forest model on the processed feature set.
        """
        if not os.path.exists(data_path):
            return {"error": "Dataset not found"}

        print(f"Loading training data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Ensure all columns exist, fill missing with 0 for safety in MVP
        for col in self.features:
            if col not in df.columns:
                df[col] = 0
        
        X = df[self.features]
        y = df['label']

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "report": classification_report(y_test, preds, output_dict=True)
        }
        
        print(f"Model Accuracy: {metrics['accuracy']:.2f}")
        
        # Save
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")
        
        return {
            "status": "success",
            "metrics": metrics,
            "model_path": self.model_path
        }

    def get_feature_importance(self):
        """
        Returns feature importance sorted by impact.
        """
        if not os.path.exists(self.model_path):
             return []
             
        try:
            model = joblib.load(self.model_path)
            if not hasattr(model, 'feature_importances_'):
                return []
                
            importances = model.feature_importances_
            feature_imp = []
            
            for i, feat in enumerate(self.features):
                if i < len(importances):
                    feature_imp.append({
                        "feature": feat,
                        "importance": float(importances[i])
                    })
            
            # Sort desc
            feature_imp.sort(key=lambda x: x['importance'], reverse=True)
            return feature_imp
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return []

    def predict_risk(self, inputs):
        """
        Predicts risk probability for a single dictionary of inputs.
        Inputs dict must match self.features keys.
        """
        if not os.path.exists(self.model_path):
             return None # Model not trained
             
        loaded_model = joblib.load(self.model_path)
        
        # Align input dictionary to feature list
        # Create DF for single sample
        df = pd.DataFrame([inputs])
        
        # Ensure columns
        for col in self.features:
            if col not in df.columns:
                df[col] = 0
                
        # Fill NA
        df = df.fillna(0)
        
        prob = loaded_model.predict_proba(df[self.features])[0][1]
        return prob
