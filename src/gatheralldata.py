
import logging
import json
import pandas as pd

# --- Logger Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("risk_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RiskAssesser")

try:
    from src.External_data_gather import ExternalRiskDataFetcher
    from src.ml_algo.fire_risk.model.fire_risk_model import FIRE_RISK_Model
    from src.ml_algo.natcat.model.nat_cat_model import NATCAT_Model
    from src.ml_algo.operational_risk.model.operational_risk_model import OPERATIONAL_RISK_Model
except ImportError:
    from External_data_gather import ExternalRiskDataFetcher
    from ml_algo.fire_risk.model.fire_risk_model import FIRE_RISK_Model
    from ml_algo.natcat.model.nat_cat_model import NATCAT_Model
    from ml_algo.operational_risk.model.operational_risk_model import OPERATIONAL_RISK_Model
    logger.warning("Modules imported from root directory (src not found).")

# Initialize models
try:
    fire_model = FIRE_RISK_Model(degree=2)
    op_model = OPERATIONAL_RISK_Model(degree=2)
    natcat_model = NATCAT_Model(degree=2)
except Exception as e:
    logger.error(f"Failed to initialize ML models: {e}")

class RiskAssessmentEngine:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        try:
            self.external_fetcher = ExternalRiskDataFetcher()
            self.all_sheets = pd.read_excel(self.excel_path, sheet_name=None)
        except Exception as e:
            logger.critical(f"Failed to load Excel file: {e}")
            raise

    def get_internal_data(self, claim_id):
        logger=logging.getLogger("InternalDataCollector")
        """Gather all internal sheets related to claim_id"""
        internal_data = {}
        logger.info(f"fetching internal data for Claim ID: {claim_id}")
        for sheet_name, df in self.all_sheets.items():
            if 'client_id' in df.columns:
                row = df[df['client_id'] == claim_id]
                if not row.empty:
                    internal_data[sheet_name] = row.iloc[0].to_dict()
        
        if not internal_data:
            logger.warning(f"No internal data found for Claim ID: {claim_id}")
        logger.info(f" Internal data fetched successfully")
        return internal_data

    def get_external_data(self, internal_location_data):
        """Fetch external data using coordinates from internal location sheet"""
        try:
            logger=logging.getLogger("ExternalDataCollector")
            logger.info(f"gathering External data ")
            if isinstance(internal_location_data, str):
                internal_location_data = dict(internal_location_data)
            
            prop_details = internal_location_data.get("property_details", {})
            if "address" not in prop_details:
                logger.error("Address missing in property_details.")
                raise ValueError("property_location data not found for this claim_id")
            
            latitude = prop_details.get("latitude")
            longitude = prop_details.get("longitude")
            address = prop_details.get("address")
            
            external_data = self.external_fetcher.get_all_external_risk_data(
                address, lat=float(latitude), lon=float(longitude)
            )
            logger.info(f" External data fetched successfully")
            return external_data
        except Exception as e:
            logger.error(f"External data fetch failed: {e}")
            return {}

    def compute_derived_metrics(self, internal_data, external_data):
        """Compute derived risk scores"""
        logger.info("Computing risk  metrics")
        
        # --- Fire Risk ---
    
        fire_internal = internal_data.get("fire_protection", {})
        fire_external = external_data.get("fire_brigade", {})
        sprinkler_score = fire_internal.get("sprinkler_coverage_pct", 0) / 100
        fire_brigade_score = max(0, 1 - fire_external.get("nearest_station_distance_km", 10)/10)
        detection_score = 1 if fire_internal.get("detection_system", None) else 0
        housekeeping_score = internal_data.get("risk_management", {}).get("housekeeping_score", 0)/5
        
        fire_risk_user_input = {
            "sprinkler_score": sprinkler_score,
            "fire_brigade_score": fire_brigade_score,
            "detection_score": detection_score,
            "housekeeping_score": housekeeping_score
        }
        
        try:
            prediction = fire_model.inference(fire_risk_user_input)
            fire_risk_score = round(prediction, 2)
        except Exception as e:
            logger.exception("Error in Fire Risk Model Inference, using fallback calculation.")
            fire_risk_score = round(0.4*sprinkler_score + 0.3*fire_brigade_score + 0.2*detection_score + 0.1*housekeeping_score, 2)

        # --- NatCat Risk ---
        flood_level_map = {"Low": 0.9, "Medium": 0.6, "High": 0.3}
        earthquake_map = {"Low": 0.9, "Medium": 0.6, "High": 0.3}
        
        flood_score = flood_level_map.get(external_data.get("flood", {}).get("flood_risk_level", "Low"), 0.6)
        earthquake_score = earthquake_map.get(external_data.get("earthquake", {}).get("seismic_risk_level", "Low"), 0.6)
        raw_wind = external_data.get("weather", {}).get("max_wind_gust_kmh")

        try:
            if raw_wind is not None and not isinstance(raw_wind, (str, bytes)):
                wind_val = float(raw_wind)
            else:
                wind_val = 50.0
        except (ValueError, TypeError):
            logger.warning("Invalid wind data; using default 50.0")
            wind_val = 50.0

        wind_score = min(wind_val / 150, 1.0)
        nat_risk_user_input = {
            "flood_score": flood_score,
            "earthquake_score": earthquake_score,
            "wind_score": wind_score
        }

        try:
            prediction = natcat_model.inference(nat_risk_user_input)
            natcat_score = round(prediction, 2)
        except Exception as e:
            logger.exception("Error in NatCat Model Inference, using fallback.")
            natcat_score = round(0.5*flood_score + 0.3*earthquake_score + 0.2*wind_score, 2)

        # --- Operational Risk ---
        site_data = internal_data.get("site_and_operations", {})
        single_site = site_data.get("criticality_to_business", "High") == "High"
        critical_process = 1 if site_data.get("criticality_to_business", "High") == "High" else 0
        bi_sum_insured = internal_data.get("account_and_exposure", {}).get("business_interruption_sum_insured", 0)
        normalized_bi = min(bi_sum_insured/1e8, 1)
        loss_flag = internal_data.get("loss_history", {}).get("loss_last_5_years", False)
        
        operational_risk_user_input = {
            "single_site": single_site,
            "critical_process": critical_process,
            "normalized_bi": normalized_bi,
            "loss_flag": loss_flag
        }
        
        try:
            prediction = op_model.inference(operational_risk_user_input)
            operational_risk_score = round(prediction, 2)
        except Exception as e:
            logger.exception("Error in Operational Risk Model Inference, using fallback.")
            operational_risk_score = round(0.3*(1-int(single_site)) + 0.3*critical_process + 0.2*normalized_bi + 0.2*(1-int(loss_flag)), 2)

        # --- Final Calculation ---
        overall_risk_score = round(0.4*fire_risk_score + 0.4*natcat_score + 0.2*operational_risk_score, 2)
        
        if overall_risk_score >= 0.85: grade = "A"
        elif overall_risk_score >= 0.70: grade = "B"
        elif overall_risk_score >= 0.65: grade = "C"
        else: grade = "D"
        logger.info("risk scoring completed ")
        logger.info(f"Assessment complete")

        return {
            "individual scores": {
                "fire_risk_score": fire_risk_score,
                "natcat_score": natcat_score,
                "operational_risk_score": operational_risk_score,
            },
            "overall_risk_score": overall_risk_score,
            "risk_grade": grade
        }

    def assess_claim(self, claim_id):
        """Main function to generate the complete JSON output for a claim"""
        internal_data = self.get_internal_data(claim_id)
        external_data = self.get_external_data(internal_data)
        derived_data = self.compute_derived_metrics(internal_data, external_data)
        
        return {
            "claim_id": claim_id,
            "internal": internal_data,
            "external": external_data,
            "derived": derived_data
        }

if __name__ == "__main__":
    # engine = RiskAssessmentEngine(r"C:\Users\Ranjith\Desktop\Risk_Engineering\data\data.xlsx")
    # result = engine.assess_claim("C001")
    pass
