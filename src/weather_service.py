import requests

class WeatherService:
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1/forecast"

    def get_current_weather(self, lat, lon):
        """
        Fetches real-time weather data from Open-Meteo.
        Variables: Temperature (2m), Rel Humidity (2m), Precip, Wind Speed (10m)
        """
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
                "wind_speed_unit": "kmh"
            }
            
            response = requests.get(self.base_url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            curr = data.get("current", {})
            
            return {
                "temp_c": curr.get("temperature_2m", 25.0),
                "humidity": curr.get("relative_humidity_2m", 50.0),
                "wind_kph": curr.get("wind_speed_10m", 10.0),
                "precip_mm": curr.get("precipitation", 0.0)
            }
        except Exception as e:
            print(f"Weather API Error: {e}")
            # Fallback to sensible defaults
            return {
                "temp_c": 25.0,
                "humidity": 50.0,
                "wind_kph": 10.0,
                "precip_mm": 0.0,
                "source": "fallback"
            }

    def get_forecast(self, lat, lon):
        """
        Fetches 7-day weather forecast from Open-Meteo.
        """
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,precipitation_sum,wind_speed_10m_max",
                "wind_speed_unit": "kmh",
                "timezone": "auto"
            }
            
            response = requests.get(self.base_url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            daily = data.get("daily", {})
            times = daily.get("time", [])
            temps = daily.get("temperature_2m_max", [])
            precips = daily.get("precipitation_sum", [])
            winds = daily.get("wind_speed_10m_max", [])
            
            forecast = []
            for i in range(len(times)):
                forecast.append({
                    "date": times[i],
                    "temp_c": temps[i],
                    "precip_mm": precips[i],
                    "wind_kph": winds[i]
                })
            return forecast
            
        except Exception as e:
            print(f"Weather Forecast API Error: {e}")
            return []
