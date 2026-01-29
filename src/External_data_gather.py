# from geopy.geocoders import Nominatim
# from meteostat import Point, Daily
# from datetime import datetime, timedelta
# import osmnx as ox
# import networkx as nx
# import feedparser
# import json
# import math
# import pandas as pd
# import warnings
# warnings.filterwarnings('ignore')
# class ExternalRiskDataFetcher:
#     def __init__(self, avg_fire_truck_speed_kmh=35):
#         self.geolocator = Nominatim(user_agent="risk_engineering_crm")
#         self.avg_speed = avg_fire_truck_speed_kmh

#     # -----------------------------
#     # 1. Geocoding
#     # -----------------------------
#     def get_location(self, address):
#         loc = self.geolocator.geocode(address)
#         if not loc:
#             return None

#         return {
#             "address": address,
#             "latitude": loc.latitude,
#             "longitude": loc.longitude,
#             "source": "OpenStreetMap",
#             "confidence": "High"
#         }

#     # -----------------------------
#     # 2. Weather & Wind
#     # -----------------------------
#     def get_weather_risk(self, lat, lon):
       
#         try:
#             point = Point(lat, lon)
#             # Fetching 3 years of daily data
#             start = datetime.now() - timedelta(days=365 * 3)
#             end = datetime.now()
#             data = Daily(point, start, end).fetch()
            
#             # If no data found, return None (NOT a string)
#             if data.empty:
#                 return {
#                     "max_wind_speed_kmh": None, 
#                     "max_wind_gust_kmh": None,
#                     "source": "Meteostat (No data found)"
#                 }

#             # Extract max values
#             wspd = data["wspd"].max()
#             wpgt = data["wpgt"].max()

#             # Ensure we only round if the value is a valid number (not NA)
#             return {
#                 "max_wind_speed_kmh": round(float(wspd), 1) if pd.notna(wspd) else None,
#                 "max_wind_gust_kmh": round(float(wpgt), 1) if pd.notna(wpgt) else None,
#                 "source": "Meteostat"
#             }
#         except Exception as e:
#             print(f"Error fetching weather: {e}")
#             # Return None on error to prevent calculation crashes
#             return {
#                 "max_wind_speed_kmh": None, 
#                 "max_wind_gust_kmh": None, 
#                 "source": "Error"
#             }
    
#     # -----------------------------
#     # 3. Earthquake Risk (USGS Feed)
#     # -----------------------------
#     def get_earthquake_risk(self, lat, lon, radius_km=300):
        
#         """
#         Uses USGS public earthquake feed (last 30 days, M>=4.5)
#         """
#         feed_url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_month.atom"
#         feed = feedparser.parse(feed_url)

#         magnitudes = []

#         for entry in feed.entries:
#             if "georss_point" not in entry:
#                 continue

#             eq_lat, eq_lon = map(float, entry.georss_point.split())
#             distance = self._haversine(lat, lon, eq_lat, eq_lon)

#             if distance <= radius_km:
#                 mag = float(entry.title.split("M")[1].split(",")[0])
#                 magnitudes.append(mag)

#         max_mag = max(magnitudes) if magnitudes else 0

#         return {
#             "events_last_30_days": len(magnitudes),
#             "max_magnitude": max_mag,
#             "radius_km": radius_km,
#             "seismic_risk_level": self._classify_seismic(max_mag),
#             "source": "USGS Earthquake Feed",
#             "confidence": "Medium"
#         }

#     def _classify_seismic(self, mag):
       
#         if mag < 5:
#             return "Low"
#         elif mag < 6.5:
#             return "Moderate"
#         else:
#             return "High"

#     # -----------------------------
#     # 4. Fire Brigade Proximity
#     # -----------------------------
#     def get_fire_brigade_risk(self, lat, lon, search_radius_m=5000):
      
#         G = ox.graph_from_point(
#             (lat, lon),
#             dist=search_radius_m,
#             network_type="drive",
#             simplify=True
#         )

#         fire_stations = ox.features_from_point(
#             (lat, lon),
#             tags={"amenity": "fire_station"},
#             dist=search_radius_m
#         )

#         if fire_stations.empty:
#             return {
#                 "fire_station_found": False,
#                 "confidence": "Low",
#                 "source": "OpenStreetMap"
#             }

#         site_node = ox.distance.nearest_nodes(G, lon, lat)
#         distances = []

#         for geom in fire_stations.geometry:
#             fs_lat = geom.centroid.y
#             fs_lon = geom.centroid.x
#             fs_node = ox.distance.nearest_nodes(G, fs_lon, fs_lat)

#             try:
#                 dist_m = nx.shortest_path_length(
#                     G, site_node, fs_node, weight="length"
#                 )
#                 distances.append(dist_m)
#             except nx.NetworkXNoPath:
#                 continue

#         if not distances:
#             return {
#                 "fire_station_found": True,
#                 "confidence": "Low",
#                 "source": "OpenStreetMap"
#             }

#         min_km = min(distances) / 1000
#         response_time = (min_km / self.avg_speed) * 60

#         return {
#             "fire_station_found": True,
#             "nearest_station_distance_km": round(min_km, 2),
#             "estimated_response_time_min": round(response_time, 1),
#             "stations_within_radius": len(distances),
#             "assumed_avg_speed_kmh": self.avg_speed,
#             "source": "OpenStreetMap (OSMnx)",
#             "confidence": "Medium"
#         }

#     # -----------------------------
#     # 5. Flood Risk (Proxy)
#     # -----------------------------
#     def get_flood_risk(self, elevation_m=50):
       
#         if elevation_m < 10:
#             risk = "High"
#         elif elevation_m < 30:
#             risk = "Medium"
#         else:
#             risk = "Low"

#         return {
#             "elevation_m": elevation_m,
#             "flood_risk_level": risk,
#             "method": "Elevation-based proxy",
#             "confidence": "Low"
#         }

#     # -----------------------------
#     # 6. MASTER WRAPPER
#     # -----------------------------
#     def get_all_external_risk_data(self, address:str="",lat:float=0.00,lon:float=0.00):
        
#         if lat ==0.00 and lon == 0.00:
#             location = self.get_location(address)
#             if not location:
#                 return {"error": "Geocoding failed"}

#             lat = location["latitude"]
#             lon = location["longitude"]
#         else:
#             location={"latitude":lat,"longitude":lon}
            
#         return {
#             # "location": location,
#             "weather": self.get_weather_risk(lat, lon),
#             "earthquake": self.get_earthquake_risk(lat, lon),
#             "fire_brigade": self.get_fire_brigade_risk(lat, lon),
#             "flood": self.get_flood_risk(),
#             "generated_at": datetime.utcnow().isoformat()
#         }

#     # -----------------------------
#     # Distance helper
#     # -----------------------------
#     def _haversine(self, lat1, lon1, lat2, lon2):
#         R = 6371
#         phi1, phi2 = math.radians(lat1), math.radians(lat2)
#         dphi = math.radians(lat2 - lat1)
#         dlambda = math.radians(lon2 - lon1)

#         a = (
#             math.sin(dphi / 2) ** 2
#             + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
#         )
#         return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))




# if __name__ == "__main__":
#     fetcher = ExternalRiskDataFetcher()

#     result = fetcher.get_all_external_risk_data(
#         "London Industrial Estate, UK"
#     )
#     print(json.dumps(result, indent=4))
#     pass


import logging
import json
import math
import pandas as pd
import warnings
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from meteostat import Point, Daily
import osmnx as ox
import networkx as nx
import feedparser

# --- Logger Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("external_data_fetcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ExternalDataCollector")

# Silence noisy third-party logs
logging.getLogger('osmnx').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

warnings.filterwarnings('ignore')

class ExternalRiskDataFetcher:
    def __init__(self, avg_fire_truck_speed_kmh=35):
        self.geolocator = Nominatim(user_agent="risk_engineering_crm")
        self.avg_speed = avg_fire_truck_speed_kmh

    # -----------------------------
    # 1. Geocoding
    # -----------------------------
    def get_location(self, address):
        logger.info("fetching location data to process")
        try:
            loc = self.geolocator.geocode(address)
            if not loc:
                logger.warning(f"Address not found: {address}")
                return None
            logger.info("location data process successfuly")
            return {
                "address": address,
                "latitude": loc.latitude,
                "longitude": loc.longitude,
                "source": "OpenStreetMap",
                "confidence": "High"
            }
       
        except Exception as e:
            logger.error(f"Geocoding error for {address}: {e}")
            return None

    # -----------------------------
    # 2. Weather & Wind
    # -----------------------------
    def get_weather_risk(self, lat, lon):
        logger.info(f"Fetching weather data ")
        try:
            point = Point(lat, lon)
            start = datetime.now() - timedelta(days=365 * 3)
            end = datetime.now()
            data = Daily(point, start, end).fetch()
            
            if data.empty:
                logger.warning(f"No weather data found for coordinates ({lat}, {lon})")
                return {
                    "max_wind_speed_kmh": None, 
                    "max_wind_gust_kmh": None,
                    "source": "Meteostat (No data found)"
                }

            wspd = data["wspd"].max()
            wpgt = data["wpgt"].max()
            logger.info(f" weather data gathered successfully ")
            return {
                "max_wind_speed_kmh": round(float(wspd), 1) if pd.notna(wspd) else None,
                "max_wind_gust_kmh": round(float(wpgt), 1) if pd.notna(wpgt) else None,
                "source": "Meteostat"
            }
        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            return {
                "max_wind_speed_kmh": None, 
                "max_wind_gust_kmh": None, 
                "source": "Error"
            }
    
    # -----------------------------
    # 3. Earthquake Risk (USGS Feed)
    # -----------------------------
    def get_earthquake_risk(self, lat, lon, radius_km=300):
        logger.info(f"Checking seismic activity within {radius_km}km of the client property ")
        try:
            feed_url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_month.atom"
            feed = feedparser.parse(feed_url)
            magnitudes = []

            for entry in feed.entries:
                if "georss_point" not in entry:
                    continue

                eq_lat, eq_lon = map(float, entry.georss_point.split())
                distance = self._haversine(lat, lon, eq_lat, eq_lon)

                if distance <= radius_km:
                    # Parsing magnitude from title e.g., "M 5.1 - 10km N of..."
                    try:
                        mag = float(entry.title.split("M")[1].split()[0].replace(',', ''))
                        magnitudes.append(mag)
                    except (IndexError, ValueError):
                        continue

            max_mag = max(magnitudes) if magnitudes else 0
            logger.info(f"seismic activity  assessed successfully")
            return {
                "events_last_30_days": len(magnitudes),
                "max_magnitude": max_mag,
                "radius_km": radius_km,
                "seismic_risk_level": self._classify_seismic(max_mag),
                "source": "USGS Earthquake Feed",
                "confidence": "Medium"
            }
        except Exception as e:
            logger.exception(f"Earthquake feed parsing failed: {e}")
            return {"error": "Seismic data unavailable"}

    def _classify_seismic(self, mag):
        if mag < 5: return "Low"
        elif mag < 6.5: return "Moderate"
        else: return "High"

    # -----------------------------
    # 4. Fire Brigade Proximity
    # -----------------------------
    def get_fire_brigade_risk(self, lat, lon, search_radius_m=5000):
        logger.info(f"Querying OpenStreetMap for fire stations near ")
        try:
            G = ox.graph_from_point((lat, lon), dist=search_radius_m, network_type="drive", simplify=True)
            fire_stations = ox.features_from_point((lat, lon), tags={"amenity": "fire_station"}, dist=search_radius_m)

            if fire_stations.empty:
                logger.warning("No fire stations found within search radius.")
                return {
                    "fire_station_found": False,
                    "confidence": "Low",
                    "source": "OpenStreetMap"
                }

            site_node = ox.distance.nearest_nodes(G, lon, lat)
            distances = []

            for geom in fire_stations.geometry:
                fs_lat = geom.centroid.y
                fs_lon = geom.centroid.x
                try:
                    fs_node = ox.distance.nearest_nodes(G, fs_lon, fs_lat)
                    dist_m = nx.shortest_path_length(G, site_node, fs_node, weight="length")
                    distances.append(dist_m)
                except Exception:
                    continue

            if not distances:
                logger.warning("Fire stations found but no driveable path exists.")
                return {"fire_station_found": True, "path_error": "No road connection", "confidence": "Low"}

            min_km = min(distances) / 1000
            response_time = (min_km / self.avg_speed) * 60
            logger.info(f" gathered  data for Nearest station ")

            return {
                "fire_station_found": True,
                "nearest_station_distance_km": round(min_km, 2),
                "estimated_response_time_min": round(response_time, 1),
                "stations_within_radius": len(distances),
                "source": "OpenStreetMap (OSMnx)",
                "confidence": "Medium"
            }
        except Exception as e:
            logger.error(f"Fire brigade risk calculation failed: {e}")
            return {"error": "Spatial analysis failed"}

    # -----------------------------
    # 5. Flood Risk (Proxy)
    # -----------------------------
    def get_flood_risk(self, elevation_m=50):
        risk = "High" if elevation_m < 10 else "Medium" if elevation_m < 30 else "Low"
        return {
            "elevation_m": elevation_m,
            "flood_risk_level": risk,
            "method": "Elevation-based proxy",
            "confidence": "Low"
        }

    # -----------------------------
    # 6. MASTER WRAPPER
    # -----------------------------
    def get_all_external_risk_data(self, address:str="", lat:float=0.00, lon:float=0.00):
        
        if lat == 0.00 and lon == 0.00:
            location = self.get_location(address)
            if not location:
                return {"error": "Geocoding failed"}
            lat, lon = location["latitude"], location["longitude"]
            
        data = {
            "weather": self.get_weather_risk(lat, lon),
            "earthquake": self.get_earthquake_risk(lat, lon),
            "fire_brigade": self.get_fire_brigade_risk(lat, lon),
            "flood": self.get_flood_risk(),
            "generated_at": datetime.utcnow().isoformat()
        }
        logger.info("Master data fetch completed successfully.")
        return data

    def _haversine(self, lat1, lon1, lat2, lon2):
        R = 6371
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = (math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
        return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

if __name__ == "__main__":
    fetcher = ExternalRiskDataFetcher()
    result = fetcher.get_all_external_risk_data("London Industrial Estate, UK")
    print(json.dumps(result, indent=4))