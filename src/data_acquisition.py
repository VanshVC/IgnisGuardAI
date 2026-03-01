import ee
import pandas as pd
import os
from datetime import datetime, timedelta
import math
import time

class HistoricalFireDataCollector:
    def __init__(self, output_dir="data/historical"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        try:
            # Re-using the project ID from satellite_pipeline.py
            ee.Initialize(project='crafty-mile-483017-r5')
        except Exception as e:
            print(f"GEE Init warning: {e}")
            try:
                ee.Initialize()
            except:
                pass

    def fetch_fire_data(self, region_coords, start_date, end_date, limit=1000, enrich=False):
        """
        Fetches historical fire data from MODIS and VIIRS via the FIRMS collection.
        Optionally enriches it with Satellite and Weather data.
        """
        region = ee.Geometry.Polygon([region_coords])
        
        # FIRMS dataset
        dataset = ee.ImageCollection('FIRMS') \
            .filterDate(start_date, end_date) \
            .filterBounds(region)

        def process_image(img):
            fire_pixels = img.select('frp').gt(0)
            img_geo = img.addBands(ee.Image.pixelLonLat())
            samples = img_geo.updateMask(fire_pixels).sample(
                region=region,
                scale=1000,
                numPixels=limit,
                geometries=True
            )
            date = img.date().format('YYYY-MM-DD')
            return samples.map(lambda f: f.set('date', date))

        flat_collection = dataset.map(process_image).flatten()
        features = flat_collection.limit(limit).getInfo()['features']
        
        data_list = []
        for f in features:
            props = f['properties']
            coords = f['geometry']['coordinates'] # [lon, lat]
            
            data_list.append({
                'latitude': coords[1],
                'longitude': coords[0],
                'date': props.get('date', 'Unknown'),
                'frp': props.get('frp', 0.0),
                'confidence': props.get('confidence', 0),
                'satellite': 'MODIS/VIIRS'
            })
            
        df = pd.DataFrame(data_list)
        
        if df.empty:
            return {"status": "success", "count": 0, "message": "No fire pixels found in region."}
            
        if enrich:
            print(f"Enriching {len(df)} data points (this may take time)...")
            df = self.enrich_data_points(df)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_enriched" if enrich else ""
        filename = f"fire_data_{start_date}_{end_date}_{timestamp}{suffix}.csv"
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        
        return {
            "status": "success",
            "count": len(df),
            "file": filepath,
            "preview": df.head().to_dict()
        }

    def enrich_data_points(self, df):
        """
        Iterates through fire points and attaches:
        1. Pre-fire Spectral Indices (NDVI, NBR, LST) from Landsat 8/9
        2. Weather conditions (Temp, Wind, Humid) from ERA5
        3. Terrain (Elevation, Slope) from SRTM
        """
        enriched_rows = []
        
        # Iterate over DataFrame
        # Note: synchronus calls in loop are slow. 
        # For production, we would map over FeatureCollection in GEE.
        # For prototype (<100 points), this is acceptable and debuggable.
        
        for idx, row in df.iterrows():
            lat, lon = row['latitude'], row['longitude']
            date_str = row['date']
            fire_date = datetime.strptime(date_str, '%Y-%m-%d')
            
            # 1. Satellite (Look back 30 days for clear pre-fire image)
            sat_start = (fire_date - timedelta(days=30)).strftime('%Y-%m-%d')
            sat_end = fire_date.strftime('%Y-%m-%d')
            
            point = ee.Geometry.Point([lon, lat])
            
            # Landsat 8 Collection
            l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
                .filterBounds(point) \
                .filterDate(sat_start, sat_end) \
                .sort('CLOUD_COVER') \
                .first()
            
            # Default values
            ndvi, nbr, lst, swir = None, None, None, None
            
            try:
                # Need to check if image exists
                # getInfo() forces the call
                img_props = l8.toDictionary().select(['system:id']).getInfo()
                
                if img_props:
                    l8_func = self._add_bands_l8(l8)
                    # Reduce at point
                    values = l8_func.reduceRegion(
                        reducer=ee.Reducer.first(),
                        geometry=point,
                        scale=30
                    ).getInfo()
                    
                    ndvi = values.get('NDVI')
                    nbr = values.get('NBR')
                    lst = values.get('LST')
                    swir = values.get('SR_B6') # SWIR 1
            except Exception as e:
                # print(f"Sat Error at {idx}: {e}")
                pass

            # 2. Weather (ERA5 Reanalysis - Daily)
            # ERA5 Land Daily Aggregated
            weather_date = fire_date.strftime('%Y-%m-%d')
            era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
                .filterDate(weather_date, (fire_date + timedelta(days=1)).strftime('%Y-%m-%d')) \
                .first()
                
            temp, humid, wind, precip = None, None, None, None
            
            try:
                w_values = era5.reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=point,
                    scale=11000 # ~11km resolution
                ).getInfo()
                
                if w_values:
                    # Temp in Kelvin to Celsius
                    t_k = w_values.get('temperature_2m_mean', 300)
                    temp = t_k - 273.15
                    
                    # Humidity Proxy from Dewpoint
                    d_k = w_values.get('dewpoint_temperature_2m_mean', 290)
                    # Simple approx: RH ~= 100 - 5 * (T - Td) (in C)
                    humid = 100 - 5 * ((t_k - 273.15) - (d_k - 273.15))
                    humid = max(0, min(100, humid))
                    
                    # Wind Speed from U and V components
                    u = w_values.get('u_component_of_wind_10m_mean', 0)
                    v = w_values.get('v_component_of_wind_10m_mean', 0)
                    wind = math.sqrt(u**2 + v**2) * 3.6 # m/s to km/h
                    
                    precip = w_values.get('total_precipitation_sum', 0) * 1000 # m to mm
            except:
                pass
            
            # 3. Terrain
            slope, elevation = None, None
            try:
                srtm = ee.Image('USGS/SRTMGL1_003')
                terrain = ee.Terrain.products(srtm)
                t_values = terrain.reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=30).getInfo()
                slope = t_values.get('slope')
                elevation = t_values.get('elevation')
            except:
                pass

            # Update Row
            row_dict = row.to_dict()
            row_dict.update({
                'NDVI': ndvi,
                'NBR': nbr,
                'LST': lst,
                'SWIR': swir,
                'Temperature_C': temp,
                'Humidity': humid,
                'Wind_Speed_kmh': wind,
                'Precipitation_mm': precip,
                'Slope': slope,
                'Elevation': elevation
            })
            enriched_rows.append(row_dict)
            
            # Rate limit politeness
            if idx % 10 == 0:
                print(f"Processed {idx+1}/{len(df)}")
        
        return pd.DataFrame(enriched_rows)

    def _add_bands_l8(self, image):
        # Landsat 8 Surface Reflectance
        # B4=Red, B5=NIR, B7=SWIR2, B10=Thermal
        
        # Scales: SR bands are 0.0000275 + -0.2 (Collection 2)
        def scale(b): 
            return image.select(b).multiply(0.0000275).add(-0.2)
        
        red = scale('SR_B4')
        nir = scale('SR_B5')
        swir2 = scale('SR_B7')
        
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
        nbr = nir.subtract(swir2).divide(nir.add(swir2)).rename('NBR')
        
        # LST: ST_B10 * 0.00341802 + 149.0
        lst = image.select('ST_B10').multiply(0.00341802).add(149.0).rename('LST')
        
        return image.addBands([ndvi, nbr, lst])

# Example Usage
if __name__ == "__main__":
    collector = HistoricalFireDataCollector()
    # Test with small region
    coords = [
         [-121.60, 39.70],
         [-121.50, 39.70],
         [-121.50, 39.80],
         [-121.60, 39.80]
    ]
    # Run small test with enrichment
    print("Fetching data with enrichment...")
    collector.fetch_fire_data(coords, '2018-11-08', '2018-11-08', limit=5, enrich=True)
