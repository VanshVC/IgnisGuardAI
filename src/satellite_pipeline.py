import ee
from datetime import datetime, timedelta

class GEESatellitePipeline:
    def __init__(self):
        ee.Initialize(project='crafty-mile-483017-r5')

    def add_indices(self, image):
        ndvi = image.normalizedDifference(
            ['SR_B5', 'SR_B4']
        ).rename('NDVI')

        nbr = image.normalizedDifference(
            ['SR_B5', 'SR_B7']
        ).rename('NBR')

        return image.addBands([ndvi, nbr])

    def add_lst(self, image):
        lst = (
            image.select('ST_B10')
            .multiply(0.00341802)
            .add(149.0)
            .rename('LST')
        )
        return image.addBands(lst)

    def detect_fire(self, image, region):
        image = self.add_indices(image)
        image = self.add_lst(image)

        fire_mask = (
            image.select('LST').gt(335)
            .And(image.select('NDVI').lt(0.25))
            .And(image.select('NBR').lt(0.05))
        )

        fire_pixels = fire_mask.selfMask()

        stats = fire_pixels.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region,
            scale=30,
            maxPixels=1e9
        ).getInfo()

        pixel_count = list(stats.values())[0] if stats else 0
        area_acres = round((pixel_count * 900) / 4046.86, 2)

        coords = ee.Image.pixelLonLat().updateMask(fire_mask).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=30,
            maxPixels=1e9
        ).getInfo()

        fire_location = None
        if coords and 'latitude' in coords:
            fire_location = {
                "lat": coords['latitude'],
                "lon": coords['longitude']
            }

        fire_type = "NONE"
        risk = "LOW"

        if area_acres > 100:
            fire_type = "FOREST_FIRE"
            risk = "HIGH"
        elif area_acres > 2:
            fire_type = "FARM_FIRE"
            risk = "MEDIUM"

        return {
            "fire_detected": pixel_count > 0,
            "pixel_count": pixel_count,
            "burned_area_acres": area_acres,
            "fire_type": fire_type,
            "risk_level": risk,
            "fire_location": fire_location,
            "method": "Thermal + NDVI + NBR (Landsat 9)"
        }

    def analyze_region_risk(self, image, region):
        """
        Extracts raw environmental risk factors (LST, NDVI, NBR, SWIR) for a region
        regardless of active fire status.
        """
        image = self.add_indices(image)
        image = self.add_lst(image)

        # Get Mean Stats
        # SR_B6 is SWIR1 for Landsat 9
        stats = image.select(['LST', 'NDVI', 'NBR', 'SR_B6']).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=30,
            maxPixels=1e9
        ).getInfo()

        return {
            "mean_lst_k": stats.get('LST', 300), 
            "mean_ndvi": stats.get('NDVI', 0.5),
            "mean_nbr": stats.get('NBR', 0.0),
            "mean_swir": stats.get('SR_B6', 0.0)
        }

    def get_risk_factors(self, coords, days=30):
        """
        Fetches the latest available satellite image and extracts risk factors.
        """
        region = ee.Geometry.Polygon([coords])
        end = datetime.now()
        start = end - timedelta(days=days)

        collection = (
            ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
            .filterBounds(region)
            .filterDate(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
            .sort('system:time_start', False)
        )

        if collection.size().getInfo() == 0:
            return None

        image = collection.first()
        return self.analyze_region_risk(image, region)

    def get_terrain_data(self, coords):
        """
        Fetches Elevation and Slope from SRTM Digital Elevation Data.
        """
        try:
            region = ee.Geometry.Polygon([coords])
            srtm = ee.Image('USGS/SRTMGL1_003')
            terrain = ee.Terrain.products(srtm)
            
            stats = terrain.select(['elevation', 'slope']).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=30,
                maxPixels=1e9
            ).getInfo()
            
            return {
                "elevation": stats.get('elevation', 0),
                "slope": stats.get('slope', 0)
            }
        except Exception as e:
            print(f"Terrain Fetch Error: {e}")
            return {"elevation": 0, "slope": 0}

    def generate_thumbnail(self, image, region):
        # Landsat 9 L2 visualization
        # SR_B4 (Red), SR_B3 (Green), SR_B2 (Blue)
        # Scaling roughly for SR range
        vis_params = {
            'bands': ['SR_B4', 'SR_B3', 'SR_B2'],
            'min': 7500,
            'max': 12000,
            'dimensions': '512x512',
            'region': region,
            'format': 'png'
        }
        try:
            return image.getThumbURL(vis_params)
        except:
            return "https://via.placeholder.com/512x288?text=Satellite+Error"
