
import math

class SpreadPredictor:
    def __init__(self):
        """
        Calculates fire spread vectors based on Rothermel's Surface Fire Spread Model (Simplified).
        """
        pass

    def calculate_spread_vector(self, wind_speed_kmh, wind_direction_deg, vegetation_density):
        """
        Returns the spread vector (distance in km over 3 hours) and target coordinates delta.
        
        Args:
            wind_speed_kmh (float): Wind speed in km/h.
            wind_direction_deg (int): Direction wind is blowing FROM (0=N, 90=E).
            vegetation_density (float): NDVI value (0.0 to 1.0).
        
        Returns:
            dict: {
                "velocity_kmh": float,
                "spread_direction": int,
                "predicted_acres_3hr": float,
                "intensity_factor": str
            }
        """
        # 1. Base Spread Rate (ROS) based on Vegetation (NDVI)
        # Higher density = more fuel = potentially slower spread but higher intensity
        # Dry/Medium density (0.2 - 0.5) spreads fastest.
        fuel_factor = 1.0
        if 0.2 <= vegetation_density <= 0.5:
            fuel_factor = 1.5  # Flashy fuels (grass/shrub)
        elif vegetation_density > 0.6:
            fuel_factor = 0.8  # Dense forest (slower but intense)

        # 2. Wind Factor (Exponential influence)
        # Empirical rule: ROS roughly doubles for every 15 km/h increase
        wind_factor = 1 + (0.05 * wind_speed_kmh ** 1.5)
        
        # 3. Calculate Rate of Spread (km/h)
        ros_kmh = 0.1 * fuel_factor * wind_factor
        
        # Spread is roughly DOWNWIND. If wind comes FROM North (0), fire goes SOUTH (180).
        spread_direction = (wind_direction_deg + 180) % 360
        
        # 4. Impact Estimation (3 Hours later)
        dist_3h = ros_kmh * 3
        # Assuming ellipse spread shape, Area ~= PI * a * b
        predicted_acres = (math.pi * (dist_3h * 1000) ** 2) / 4046.86 / 4 # Rough approximation
        
        # 5. Intensity Classification
        intensity = "MODERATE"
        if ros_kmh > 1.5: strength = "EXTREME FLANKING"
        elif ros_kmh > 0.5: strength = "HIGH VELOCITY"
        else: strength = "CREEPING"

        return {
            "velocity_kmh": round(ros_kmh, 2),
            "spread_direction": spread_direction,
            "predicted_dist_3h_km": round(dist_3h, 2),
            "predicted_acres_3hr": round(predicted_acres, 2),
            "threat_status": strength
        }
