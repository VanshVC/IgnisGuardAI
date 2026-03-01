from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
from src.detector import FireDetector
from src.satellite_pipeline import GEESatellitePipeline
from src.detector import FireDetector
from src.satellite_pipeline import GEESatellitePipeline
from src.optimizer import PrecisionOptimizer
from src.train import FireTrainer
from src.decision_engine import DecisionEngine
from src.analytics import FireAnalytics
from src.pred.spread_model import SpreadPredictor
from src.pred.risk_model import FireRiskPredictor
from src.data_acquisition import HistoricalFireDataCollector
from src.feature_engineering import FireFeaturePipeline
from src.pred.prediction_model import FirePredictionTrainer
from src.weather_service import WeatherService
from src.alert_system import AlertManager
import ee

app = FastAPI(title="IgnisGuard AI API")

# Initialize engines
detector = FireDetector()
optimizer = PrecisionOptimizer()
decision_engine = DecisionEngine(confidence_threshold=0.75)
analytics_engine = FireAnalytics()
spread_engine = SpreadPredictor()
risk_predictor = FireRiskPredictor()
risk_predictor = FireRiskPredictor()
data_collector = HistoricalFireDataCollector()
feature_pipeline = FireFeaturePipeline()
prediction_trainer = FirePredictionTrainer()
weather_service = WeatherService()
alert_manager = AlertManager()
try:
    satellite_engine = GEESatellitePipeline()
except Exception as e:
    print(f"GEE Engine bypass: {e}")
    satellite_engine = None

# Mount static and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = FireDetector()

# Directories for different types of uploads
UPLOAD_DIRS = {
    "satellite": "data/uploads/satellite",
    "drone": "data/uploads/drone",
    "ground": "data/uploads/ground"
}

for folder in UPLOAD_DIRS.values():
    os.makedirs(folder, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/prediction", response_class=HTMLResponse)
async def prediction_page(request: Request):
    return templates.TemplateResponse("prediction.html", {"request": request})

@app.get("/status")
async def get_status():
    return {
        "project": "IgnisGuard AI",
        "status": "Online",
        "endpoints": ["/detect/ground", "/detect/drone", "/detect/satellite", "/live-satellite"]
    }

@app.post("/detect/{input_type}")
async def detect_fire(input_type: str, file: UploadFile = File(...)):
    if input_type not in UPLOAD_DIRS:
        return {"error": "Invalid input type. Use 'satellite', 'drone', or 'ground'."}
    
    file_path = os.path.join(UPLOAD_DIRS[input_type], file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Drone videos require frame-by-frame analysis with Temporal Verification
    if file.filename.endswith(('.mp4', '.avi', '.mov')):
        output_path, is_verified = detector.process_video(file_path, decision_engine=decision_engine)
        return {
            "source_type": input_type,
            "filename": file.filename,
            "processed_video": output_path,
            "fire_detected": is_verified,
            "alert_level": "CRITICAL_VERIFIED" if is_verified else "NORMAL"
        }

    # Perform vision detection
    results = detector.detect(file_path)
    
    # Optional: Get satellite context if available (placeholder coords)
    sat_risk = None
    if satellite_engine:
        # Mocking lat/lon for the upload if not provided - in prod, EXIF data would be used
        # Assume Farm Fire risk profile for testing differentiation
        sat_risk = {
            "risk_level": "MEDIUM", 
            "fire_type": "FARM_FIRE",
            "estimated_burn_acres": 5.0,
            "fire_location": {"lat": 18.5204, "lon": 73.8567}, # Mock GPS derived from EXIF
            "environmental_factors": {
                "temperature": 38.5,
                "humidity": 22.0,
                "spread_risk": "HIGH"
            }
        } 
    
    # Intelligent Decision Fusion
    verification = decision_engine.verify_detection(results, satellite_risk=sat_risk)
            
    return {
        "source_type": input_type,
        "filename": file.filename,
        "detections": results,
        "verification": verification,
        "fire_detected": verification["is_verified"],
        "alert_level": verification["alert_status"],
        # Pass context up for UI
        "risk_context": sat_risk
    }

@app.get("/live-satellite")
async def get_live_satellite(lat: float = 18.5204, lon: float = 73.8567, scan_mode: str = "local"):
    """
    Trigger live satellite fetch via GEE or fallback to Mock Data.
    scan_mode: 'local' or 'global'
    """
    # 0. Global Mode Logic (Randomly picks a location to simulate global scanning)
    location_name = "Local Scan"
    
    if scan_mode == "global":
        import random
        # defined Specific Hotspots for "Exact Location" simulation
        # These represent known fire-prone zones with precise names
        specific_locations = [
            {"name": "Mendocino Complex, CA, USA", "lat": 39.25, "lon": -123.12},
            {"name": "Paradise (Camp Fire Site), CA, USA", "lat": 39.76, "lon": -121.60},
            {"name": "Yosemite National Park, CA, USA", "lat": 37.86, "lon": -119.53},
            {"name": "Sequoia National Park, CA, USA", "lat": 36.48, "lon": -118.56},
            {"name": "Redding (Carr Fire Site), CA, USA", "lat": 40.58, "lon": -122.39},
            
            # South America (Amazon & Pantanal)
            {"name": "Amazonas (Deep Rainforest), Brazil", "lat": -3.46, "lon": -62.21},
            {"name": "Mato Grosso (Agri-Belt), Brazil", "lat": -12.60, "lon": -55.60},
            {"name": "Pantanal Wetlands, Brazil", "lat": -17.76, "lon": -57.65},
            {"name": "Valparaíso Region, Chile", "lat": -33.04, "lon": -71.62},
            {"name": "Gran Chaco, Argentina", "lat": -25.00, "lon": -60.00},

            # Europe
            {"name": "Black Forest, Germany", "lat": 48.20, "lon": 8.20},
            {"name": "Gironde Region, France", "lat": 44.50, "lon": -0.50},
            {"name": "Evia Island, Greece", "lat": 38.55, "lon": 23.85},
            {"name": "Antalya (Manavgat), Turkey", "lat": 36.90, "lon": 31.45},
            {"name": "Algarve Region, Portugal", "lat": 37.20, "lon": -8.10},
            {"name": "Sicily (Mount Etna), Italy", "lat": 37.75, "lon": 14.99},

            # Australia & Oceania
            {"name": "Blue Mountains, NSW, Australia", "lat": -33.71, "lon": 150.31},
            {"name": "Kangaroo Island, Australia", "lat": -35.80, "lon": 137.20},
            {"name": "Gippsland, Victoria, Australia", "lat": -37.50, "lon": 148.00},
            {"name": "Tasmanian Wilderness, Australia", "lat": -42.00, "lon": 146.00},

            # North Asia (Siberia)
            {"name": "Siberian Taiga (Yakutia), Russia", "lat": 63.50, "lon": 125.00},
            {"name": "Lake Baikal Region, Russia", "lat": 53.50, "lon": 108.00},
            {"name": "Amur Region, Russia", "lat": 51.50, "lon": 128.00},

            # Southeast Asia
            {"name": "Chiang Mai (Burn Zone), Thailand", "lat": 18.70, "lon": 98.98},
            {"name": "Sumatra (Peatland), Indonesia", "lat": -0.50, "lon": 101.50},
            {"name": "Kalimantan, Indonesia", "lat": -1.50, "lon": 113.50},

            # Africa
            {"name": "Congo Basin, DRC", "lat": -1.00, "lon": 22.00},
            {"name": "Kruger National Park, South Africa", "lat": -24.00, "lon": 31.50},
            {"name": "Madagascar Dry Forests", "lat": -19.00, "lon": 45.00},

            # India & South Asia
            {"name": "Western Ghats (Sahyadri), India", "lat": 14.30, "lon": 74.80},
            {"name": "Similipal Forest, Odisha, India", "lat": 21.92, "lon": 86.42},
            {"name": "Bandipur Tiger Reserve, India", "lat": 11.66, "lon": 76.63},
            {"name": "Jim Corbett National Park, India", "lat": 29.53, "lon": 78.77},
            {"name": "Kaziranga (Grasslands), India", "lat": 26.57, "lon": 93.17},

            # North America (Canada/USA)
            {"name": "Alberta Boreal Forest, Canada", "lat": 56.10, "lon": -111.40},
            {"name": "British Columbia (Lytton), Canada", "lat": 50.23, "lon": -121.58},
            {"name": "Yellowknife (NWT), Canada", "lat": 62.45, "lon": -114.37},
            {"name": "Yellowstone National Park, USA", "lat": 44.42, "lon": -110.58}
        ]
        
        # Weighted random choice to ensure we see both types
        # 70% Forest, 30% Farm to enable differentiation testing
        target = random.choice(specific_locations)
        location_name = target["name"]
        
        # Add small jitter so it's not the exact same pixel every time (approx 5-10km variance)
        lat_jitter = random.uniform(-0.05, 0.05)
        lon_jitter = random.uniform(-0.05, 0.05)
        
        lat = float(target["lat"] + lat_jitter)
        lon = float(target["lon"] + lon_jitter)
        
        print(f"🌍 Global Scan Initiated: Jumping to {location_name} ({lat:.4f}, {lon:.4f})")

    # 1. Try GEE Fetch (if engine exists)
    if satellite_engine:
        try:
            # Create small bounding box around coords
            offset = 0.05
            coords = [
                [lon-offset, lat-offset],
                [lon+offset, lat-offset],
                [lon+offset, lat+offset],
                [lon-offset, lat+offset]
            ]
            print(f"📡 Logic: Pulse fetching for centered region: {lat}, {lon}")
            # Use new 'run' method
            data = satellite_engine.run(coords=coords)
            
            if data and "error" not in data:
                report = data.get('fire_report', {})
                # Map new structure to old frontend expectations
                risk_analysis = {
                    "risk_level": report.get('risk_level', 'LOW'),
                    "estimated_burn_acres": report.get('burned_area_acres', 0),
                    "fire_type": report.get('fire_type', 'NONE'),
                    "environmental_factors": {
                        "temperature": 35.0, # Est
                        "humidity": 20.0     # Est
                    },
                    "high_risk_pixel_count": report.get('pixel_count', 0)
                }
                
                return {
                    "location": {"lat": lat, "lon": lon, "name": location_name},
                    "satellite_data": data,
                    "risk_analysis": risk_analysis,
                    "visual_asset": data.get('visual_asset'),
                    "fire_detected": report.get('fire_detected', False),
                    "acquisition_date": data.get('acquisition_date', "Unknown"),
                    "status": "success"
                }
        except Exception as e:
            print(f"❌ GEE Error (Falling back to mock): {e}")

    # 2. Mock Fallback (for Demo/unauthenticated usage)
    import random
    import math

    def latlon_to_tile(lat, lon, zoom):
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return xtile, ytile

    print("⚠️ Using Mock Satellite Data")
    
    # Simulate meaningful data scenarios
    rand_val = random.random()
    
    # Wind Modeling for Spread Prediction (Phase 13)
    # Speed in km/h, Direction in degrees (0=N, 90=E, 180=S, 270=W)
    wind_speed = round(random.uniform(5, 45), 1) 
    wind_direction = random.randint(0, 360) 

    # --- ADJUSTED MOCK LOGIC: Tuned for Balanced Testing ---
    # Now: 15% Chance Forest Fire, 15% Chance Farm Fire (30% Total Fire Chance)
    # This ensures "Real Time" detection is felt by the user during demos.
    if rand_val > 0.85:
        # Scenario: Massive Forest Fire (Rare Critical Event)
        acres = round(random.uniform(25, 500), 2)
        risk_level = "HIGH"
        fire_type = "FOREST_FIRE"
        is_fire_active = True
    elif rand_val > 0.70:
         # Scenario: Farm Stubble Burning (Common Agricultural Activity)
        acres = round(random.uniform(1, 15), 2)
        risk_level = "MEDIUM" # Ignored by main alert
        fire_type = "FARM_FIRE"
        is_fire_active = True 
    else:
        # Scenario: No Fire (Normal State)
        acres = 0
        risk_level = "LOW"
        fire_type = "NONE"
        is_fire_active = False

    # Spread Interpretation (Mock NDVI ~ 0.3 for dry grass)
    spread_data = spread_engine.calculate_spread_vector(
        wind_speed_kmh=wind_speed,
        wind_direction_deg=wind_direction,
        vegetation_density=0.3
    )

    # Dynamic Tile Generation for "Real" Visuals (Mocking GEE)
    # Uses Esri World Imagery (Same as frontend map) to ensure visual consistency
    zoom_level = 13
    xtile, ytile = latlon_to_tile(lat, lon, zoom_level)
    mock_image = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom_level}/{ytile}/{xtile}"

    return {
        "location": {"lat": lat, "lon": lon, "name": location_name},
        "satellite_data": {"source": "Simulated (Mock)"},
        "risk_analysis": {
            "risk_level": risk_level,
            "estimated_burn_acres": acres,
            "high_risk_pixel_count": int(acres * 10),
            "fire_type": fire_type,
            "fire_location": {"lat": lat, "lon": lon}, # Centroid of the mock fire
            "environmental_factors": {
                "temperature": round((40.2 if is_fire_active else 25.0) + random.uniform(-1.5, 1.5), 1), # Hotter if fire
                "humidity": round((18.5 if is_fire_active else 55.0) + random.uniform(-3.0, 3.0), 1),    # Drier if fire
                "spread_risk": "CRITICAL" if is_fire_active else "LOW"
            },
            "wind_data": {
                "speed_kmh": wind_speed,
                "direction_deg": wind_direction
            },
            "spread_prediction": spread_data
        },
        "visual_asset": mock_image, 
        "fire_detected": is_fire_active,
        "status": "simulated"
    }

@app.get("/predict-risk")
async def predict_risk(lat: float, lon: float):
    """
    Predicts fire risk probability for a specific location using 
    multi-source geospatial data (Satellite LST/NDVI + Weather).
    """
    # 1. Fetch Real-Time Weather (Open-Meteo)
    weather = weather_service.get_current_weather(lat, lon)
    
    humidity = weather['humidity']
    wind_speed = weather['wind_kph']
    precip = weather['precip_mm']
    
    # 2. Fetch Satellite Data (Landsat 9)
    # Defaults
    lst_k = (weather['temp_c'] + 273.15) # Fallback to air temp if sat fail
    ndvi = 0.3
    nbr = 0.0
    swir = 0.0
    
    # 3. Fetch Terrain
    slope = 0
    elevation = 0
    
    data_source_tag = "Weather Model Only"

    if satellite_engine:
        try:
             # Create small bounding box
            offset = 0.05
            coords = [
                [lon-offset, lat-offset],
                [lon+offset, lat-offset],
                [lon+offset, lat+offset],
                [lon-offset, lat+offset]
            ]
            
            # Geo-Factors
            factors = satellite_engine.get_risk_factors(coords)
            if factors:
                lst_k = factors['mean_lst_k']
                ndvi = factors['mean_ndvi']
                nbr = factors['mean_nbr']
                swir = factors['mean_swir']
                data_source_tag = "Landsat 9 + Weather Model"
                
            # Terrain
            terrain = satellite_engine.get_terrain_data(coords)
            if terrain:
                slope = terrain['slope']
                elevation = terrain['elevation']
                
        except Exception as e:
            print(f"GEE Risk Factor Error: {e}")

    prediction = risk_predictor.calculate_risk_score(
        lst_k=lst_k,
        ndvi=ndvi,
        humidity=humidity,
        wind_speed=wind_speed,
        nbr=nbr,
        swir=swir,
        precip=precip,
        slope=slope,
        elevation=elevation
    )
    
    # 4. Generate 7-Day Forecast
    forecast_data = []
    daily_forecast = weather_service.get_current_weather(lat, lon) # Default fallback
    
    try:
        raw_forecast = weather_service.get_forecast(lat, lon)
        for day in raw_forecast:
            # Approximate parameters for future days
            # LST approx: Air Temp + offset
            f_lst_k = day['temp_c'] + 273.15 + (lst_k - (weather['temp_c'] + 273.15))
            
            # Run Prediction
            f_pred = risk_predictor.calculate_risk_score(
                lst_k=f_lst_k,
                ndvi=ndvi,
                humidity=humidity, # Assuming persistence for MVP
                wind_speed=day['wind_kph'],
                nbr=nbr,
                swir=swir,
                precip=day['precip_mm'],
                slope=slope,
                elevation=elevation
            )
            
            forecast_data.append({
                "date": day['date'],
                "risk_probability": f_pred['probability'],
                "temp_c": day['temp_c'],
                "precip_mm": day['precip_mm']
            })
    except Exception as e:
        print(f"Forecast Error: {e}")

    # 5. Alert Check
    alert_status = alert_manager.evaluate_risk(
        risk_score=prediction['probability'], 
        wind_speed=wind_speed,
        location={"lat": lat, "lon": lon}
    )

    return {
        "location": {"lat": lat, "lon": lon},
        "prediction": prediction,
        "forecast": forecast_data,
        "alert": alert_status,
        "input_data": {
            "lst_k": round(lst_k, 1),
            "ndvi": round(ndvi, 2),
            "humidity": round(humidity, 1),
            "wind_speed": round(wind_speed, 1),
            "precip": precip,
            "slope": round(slope, 1)
        },
        "data_source": data_source_tag
    }


@app.post("/data/acquire")
async def acquire_historical_data(
    lat: float, 
    lon: float, 
    start_date: str = "2023-01-01", 
    end_date: str = "2023-12-31",
    enrich: bool = False
):
    """
    Triggers acquisition of historical fire data (MODIS/VIIRS) for a region.
    Creates a reproducible dataset in data/historical.
    Optionally enriches with Satellite/Weather/Terrain data.
    """
    # Create 0.5 degree bounding box (approx 50km radius)
    offset = 0.5
    region_coords = [
        [lon - offset, lat - offset],
        [lon + offset, lat - offset],
        [lon + offset, lat + offset],
        [lon - offset, lat + offset]
    ]
    
    try:
        result = data_collector.fetch_fire_data(region_coords, start_date, end_date, enrich=enrich)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/features/process")
async def process_features(file_path: str = None):
    """
    Transforms raw historical data into ML-ready features.
    If no file_path provided, uses the most recent CSV in data/historical.
    """
    if not file_path:
        # Find latest
        import glob
        list_of_files = glob.glob('data/historical/*.csv') 
        if not list_of_files:
             return {"status": "error", "message": "No historical data found to process."}
        file_path = max(list_of_files, key=os.path.getctime)
    
    try:
        result = feature_pipeline.run(file_path)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/model/train")
async def train_prediction_model(data_path: str = None):
    """
    Trains the Predictive Risk Model (RF/XGB) using the latest feature dataset.
    Saves the model as 'fire_risk_model.pkl'.
    """
    if not data_path:
        # Find latest processed feature file
        import glob
        list_of_files = glob.glob('data/processed/*.csv')
        if not list_of_files:
             return {"status": "error", "message": "No feature datasets found in data/processed."}
        data_path = max(list_of_files, key=os.path.getctime)
    
    try:
        result = prediction_trainer.train(data_path)
        result['feature_importance'] = prediction_trainer.get_feature_importance()
        return result
    except Exception as e:
         return {"status": "error", "message": str(e)}

@app.get("/model/importance")
async def get_feature_importance():
    """Returns the feature importance ranking of the current trained model."""
    return prediction_trainer.get_feature_importance()

@app.get("/alerts/recent")
async def get_recent_alerts():
    """Returns the most recent critical fire alerts logged by the system."""
    return alert_manager.get_recent_alerts()

@app.post("/feedback")
async def submit_feedback(filename: str, source_type: str, is_fire: bool):
    """
    Continuous Learning Endpoint:
    Users flag incorrect detections to improve the 'Fire Brain'.
    """
    success, message = optimizer.move_feedback_sample(filename, source_type, is_fire)
    if success:
        return {"status": "success", "message": message}
    return {"status": "error", "message": message}

@app.post("/retrain")
async def trigger_retrain():
    """
    Triggers model retraining with newly integrated 'Hard Examples'.
    Uses the upgraded YOLOv8-Medium 'High Recall' configuration.
    """
    DATA_PATH = os.path.abspath('data/processed')
    # Update to Medium model for better precision/recall balance
    trainer = FireTrainer(data_dir=DATA_PATH, model_type='yolov8m-cls.pt')
    
    # Background training to avoid blocking the API
    import threading
    # Use new defaults (150 epochs)
    thread = threading.Thread(target=trainer.train, kwargs={"epochs": 150})
    thread.start()
    return {"status": "Retraining started in background (High Recall Mode - YOLOv8m)...", "epochs": 150}

@app.get("/metrics")
async def get_metrics():
    """
    Returns live model precision, recall, mAP and accuracy metrics.
    """
    return optimizer.get_metrics_report()

@app.get("/analytics/plots")
async def get_analytics_plots():
    """
    Returns advanced visualization charts (Phase 10 & 11).
    """
    training_plot = analytics_engine.generate_training_plots()
    risk_heatmap = analytics_engine.generate_heatmap_placeholder()
    comparison_plot = analytics_engine.generate_comparison_plot()
    return {
        "training_accuracy_plot": training_plot,
        "risk_heatmap": risk_heatmap,
        "comparison_plot": comparison_plot
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)