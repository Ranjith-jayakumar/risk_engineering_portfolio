# Risk Engineering - Developer Documentation

## 📖 Complete Technical Guide

This document provides comprehensive technical details for developers working with the Risk Engineering platform. It covers architecture, implementation details, API specifications, and development workflows.

---

## Table of Contents

1. [Architecture & Design Patterns](#architecture--design-patterns)
2. [Module Specifications](#module-specifications)
3. [Machine Learning Models](#machine-learning-models)
4. [Data Processing Pipeline](#data-processing-pipeline)
5. [External Data Integration](#external-data-integration)
6. [Frontend Architecture](#frontend-architecture)
7. [API & Integration Points](#api--integration-points)
8. [Development Workflow](#development-workflow)
9. [Troubleshooting & Debugging](#troubleshooting--debugging)
10. [Performance & Optimization](#performance--optimization)

---

## Architecture & Design Patterns

### System Design Principles

The Risk Engineering platform follows these core architectural principles:

#### 1. **Separation of Concerns**
- **Data Layer**: Excel-based storage with pandas processing
- **Logic Layer**: Risk assessment engine and ML models
- **Presentation Layer**: Streamlit web interface
- **Integration Layer**: External APIs and LLM services

#### 2. **Pipeline Architecture**
```
Data Ingestion → Data Processing → Risk Computation → AI Enhancement → Report Generation
```

Each stage is independent and can be tested/optimized separately.

#### 3. **Layered Model Processing**
```
Internal Data (Excel)
        ↓
External Data (APIs)
        ↓
Feature Engineering
        ↓
ML Model Inference
        ↓
Risk Score Aggregation
        ↓
Grade Assignment
```

#### 4. **Caching Strategy**
- **Model Cache**: JSON-based model metrics (1-day TTL)
- **AI Cache**: Streamlit session state for summaries
- **Data Cache**: pandas DataFrame caching with `@st.cache_data`

### Design Patterns Used

**Factory Pattern**: Model initialization with configurable paths
```python
fire_model = FIRE_RISK_Model(degree=2)  # Creates model with configuration
```

**Strategy Pattern**: Multiple risk assessment strategies
```python
# Different models for different risk types
fire_model.inference()      # Fire risk strategy
natcat_model.inference()    # NatCat risk strategy
op_model.inference()        # Operational risk strategy
```

**Observer Pattern**: Logging system with StreamlitUIHandler
```python
# Handlers observe logger events and emit to UI
ui_handler = StreamlitUIHandler(container)
root_logger.addHandler(ui_handler)
```

**Template Method Pattern**: Risk assessment workflow
```python
def assess_claim(self, claim_id):
    # Template: fetch → compute → aggregate → return
    internal_data = self.get_internal_data(claim_id)
    external_data = self.get_external_data(internal_data)
    derived_data = self.compute_derived_metrics(...)
    return self._format_output()
```

---

## Module Specifications

### Core Modules Overview

```
Risk_Engineering/
├── frontend.py                          # Main Streamlit application
├── src/
│   ├── gatheralldata.py                 # Risk Assessment Engine
│   ├── External_data_gather.py           # External data fetcher
│   ├── gatheralldata2.py                 # Alternative implementation
│   └── ml_algo/                          # Machine learning models
│       ├── fire_risk/
│       │   ├── model/
│       │   │   └── fire_risk_model.py
│       │   └── data/
│       │       ├── fire_risk_dataset.csv
│       │       └── fire_risk_cache.json
│       ├── natcat/
│       │   ├── model/
│       │   │   └── nat_cat_model.py
│       │   └── data/
│       │       ├── natcat_dataset.csv
│       │       └── natcat_cache.json
│       └── operational_risk/
│           ├── model/
│           │   └── operational_risk_model.py
│           └── data/
│               ├── operational_risk_dataset.csv
│               └── operational_risk_cache.json
├── AI_agents/                           # AI orchestration
│   ├── risk_agent_tools.py              # Agent tools
│   └── risk_engineering_orchestrator.py # Orchestration logic
├── data/                                # Sample data
├── cache/                               # Runtime caches
└── requirements.txt                     # Dependencies
```

### Key Classes & Methods

#### 1. **RiskAssessmentEngine** (`src/gatheralldata.py`)

**Constructor:**
```python
def __init__(self, excel_path):
    """
    Initialize the Risk Assessment Engine.
    
    Args:
        excel_path (str): Path to Excel workbook with claim data
        
    Raises:
        Exception: If Excel file cannot be loaded
    """
    self.excel_path = excel_path
    self.external_fetcher = ExternalRiskDataFetcher()
    self.all_sheets = pd.read_excel(excel_path, sheet_name=None)
```

**Core Methods:**

```python
def assess_claim(self, claim_id: str) -> dict:
    """
    Main assessment method - orchestrates complete workflow.
    
    Args:
        claim_id (str): Unique claim identifier
        
    Returns:
        dict: Complete assessment with internal, external, and derived data
        
    Flow:
        1. Fetch internal data from Excel sheets
        2. Extract property coordinates
        3. Fetch external risk data
        4. Compute derived risk metrics using ML models
        5. Aggregate and return complete assessment
    """
    internal_data = self.get_internal_data(claim_id)
    external_data = self.get_external_data(internal_data)
    derived_data = self.compute_derived_metrics(internal_data, external_data)
    
    return {
        "claim_id": claim_id,
        "internal": internal_data,
        "external": external_data,
        "derived": derived_data
    }
```

```python
def get_internal_data(self, claim_id: str) -> dict:
    """
    Gather all internal claim data from Excel sheets.
    
    Args:
        claim_id (str): Client identifier
        
    Returns:
        dict: Keyed by sheet name with row data as nested dicts
        
    Implementation:
        - Iterates through all Excel sheets
        - Filters for rows matching the claim_id
        - Returns first matching row for each sheet
        - Logs warnings if no data found
    """
    internal_data = {}
    logger = logging.getLogger("InternalDataCollector")
    logger.info(f"Fetching internal data for Claim ID: {claim_id}")
    
    for sheet_name, df in self.all_sheets.items():
        if 'client_id' in df.columns:
            row = df[df['client_id'] == claim_id]
            if not row.empty:
                internal_data[sheet_name] = row.iloc[0].to_dict()
    
    if not internal_data:
        logger.warning(f"No internal data found for Claim ID: {claim_id}")
    
    logger.info("Internal data fetched successfully")
    return internal_data
```

```python
def compute_derived_metrics(self, internal_data: dict, external_data: dict) -> dict:
    """
    Calculate risk scores using ML models.
    
    Args:
        internal_data (dict): Internal claim characteristics
        external_data (dict): External environmental factors
        
    Returns:
        dict: Risk scores and final grade
        
    Calculation Flow:
        1. Extract fire risk features and call fire_model.inference()
        2. Extract NatCat features and call natcat_model.inference()
        3. Extract operational risk features and call op_model.inference()
        4. Compute weighted average: 0.4*fire + 0.4*natcat + 0.2*operational
        5. Map score to grade (A/B/C/D)
        6. Return individual and overall scores
        
    Fallback Behavior:
        - If ML model fails, uses weighted linear fallback
        - Logs error but continues with fallback calculation
    """
    logger.info("Computing risk metrics")
    
    # Fire Risk Calculation
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
        fire_risk_score = round(fire_model.inference(fire_risk_user_input), 2)
    except Exception as e:
        logger.exception("Fire Risk Model inference failed, using fallback")
        fire_risk_score = round(
            0.4*sprinkler_score + 0.3*fire_brigade_score + 
            0.2*detection_score + 0.1*housekeeping_score, 2
        )
    
    # Similar pattern for NatCat and Operational risks...
    
    # Final Aggregation
    overall_risk_score = round(
        0.4*fire_risk_score + 0.4*natcat_score + 0.2*operational_risk_score, 2
    )
    
    if overall_risk_score >= 0.85:
        grade = "A"
    elif overall_risk_score >= 0.70:
        grade = "B"
    elif overall_risk_score >= 0.65:
        grade = "C"
    else:
        grade = "D"
    
    return {
        "individual scores": {
            "fire_risk_score": fire_risk_score,
            "natcat_score": natcat_score,
            "operational_risk_score": operational_risk_score,
        },
        "overall_risk_score": overall_risk_score,
        "risk_grade": grade
    }
```

---

## Machine Learning Models

### Model Architecture

All three risk models follow the same pattern:

```python
class RiskModel:
    """
    Generic risk model using polynomial regression.
    
    Architecture:
        Input Features → PolynomialFeatures (degree 2) → LinearRegression → Score
    
    Why Polynomial Degree 2?
        - Captures non-linear relationships between features
        - Balances complexity and interpretability
        - Reduces overfitting compared to higher degrees
        - Standard for insurance risk modeling
    """
    
    def __init__(self, csv_path, cache_path, degree=2):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree)
        self.model = LinearRegression()
        
        # Load and train on dataset
        df = pd.read_csv(csv_path)
        X = df[feature_columns]
        y = df['target_score']
        
        # Train-test split with reproducibility
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Fit polynomial transformer and model
        self.poly.fit(X_train)
        X_train_poly = self.poly.transform(X_train)
        X_test_poly = self.poly.transform(X_test)
        
        self.model.fit(X_train_poly, y_train)
        
        # Calculate metrics
        y_pred = self.model.predict(X_test_poly)
        self.mse = mean_squared_error(y_test, y_pred)
        self.mae = mean_absolute_error(y_test, y_pred)
        self.r2 = r2_score(y_test, y_pred)
        
        # Cache management
        self._manage_cache()
    
    def inference(self, user_input: dict) -> float:
        """
        Predict risk score for given features.
        
        Args:
            user_input (dict): Feature dictionary matching training features
            
        Returns:
            float: Predicted risk score (0-1 scale)
        """
        X_new = pd.DataFrame([user_input])
        X_new_poly = self.poly.transform(X_new)
        prediction = self.model.predict(X_new_poly)[0]
        return round(prediction, 3)
```

### Fire Risk Model (`fire_risk_model.py`)

**Purpose**: Assess fire exposure and loss potential

**Features & Scoring**:
| Feature | Range | Source | Interpretation |
|---------|-------|--------|-----------------|
| `sprinkler_score` | 0-1 | Internal (fire_protection) | % coverage ÷ 100 |
| `fire_brigade_score` | 0-1 | External (fire_brigade) | 1 - (distance_km / 10) |
| `detection_score` | 0-1 | Internal (fire_protection) | 1 if system present, 0 otherwise |
| `housekeeping_score` | 0-1 | Internal (risk_management) | Score ÷ 5 |

**Model Performance**:
- Dataset: `fire_risk_dataset.csv` (training data)
- Algorithm: Polynomial Regression (degree 2)
- Outputs: MSE, MAE, R² metrics
- Output Scale: 0 (best) to 1 (worst)

**Example Usage**:
```python
fire_model = FIRE_RISK_Model(degree=2)

user_input = {
    "sprinkler_score": 0.8,      # 80% sprinkler coverage
    "fire_brigade_score": 0.7,    # 3km from nearest station
    "detection_score": 1,          # Has detection system
    "housekeeping_score": 0.6      # Housekeeping score of 3/5
}

risk_score = fire_model.inference(user_input)  # Returns e.g., 0.412
```

### Natural Catastrophe Model (`nat_cat_model.py`)

**Purpose**: Assess exposure to natural disasters (flood, earthquake, wind)

**Features & Scoring**:
| Feature | Range | Source | Interpretation |
|---------|-------|--------|-----------------|
| `flood_score` | 0-1 | External (flood data) | Mapped from risk level |
| `earthquake_score` | 0-1 | External (seismic data) | Mapped from magnitude |
| `wind_score` | 0-1 | External (weather data) | gust_kmh ÷ 150 |

**Risk Level Mapping**:
```python
flood_level_map = {
    "Low": 0.9,      # Low risk = high score
    "Medium": 0.6,
    "High": 0.3      # High risk = low score
}
```

**Example Usage**:
```python
natcat_model = NATCAT_Model(degree=2)

user_input = {
    "flood_score": 0.6,        # Medium flood risk
    "earthquake_score": 0.9,   # Low seismic risk
    "wind_score": 0.45         # 67.5 km/h max wind
}

risk_score = natcat_model.inference(user_input)  # Returns e.g., 0.583
```

### Operational Risk Model (`operational_risk_model.py`)

**Purpose**: Assess business interruption and operational resilience

**Features & Scoring**:
| Feature | Type | Source | Interpretation |
|---------|------|--------|-----------------|
| `single_site` | Boolean | Internal (site_and_operations) | 1 if critical to business |
| `critical_process` | Boolean | Internal (site_and_operations) | 1 if critical process present |
| `normalized_bi` | 0-1 | Internal (account_and_exposure) | BI_sum_insured ÷ 100M |
| `loss_flag` | Boolean | Internal (loss_history) | 1 if losses in last 5 years |

**Example Usage**:
```python
op_model = OPERATIONAL_RISK_Model(degree=2)

user_input = {
    "single_site": True,        # Single critical location
    "critical_process": True,   # Critical manufacturing process
    "normalized_bi": 0.75,      # $75M business interruption coverage
    "loss_flag": False          # No losses in last 5 years
}

risk_score = op_model.inference(user_input)  # Returns e.g., 0.285
```

### Cache Management

**Cache File Structure** (JSON):
```json
{
  "timestamp": "2026-01-29T14:30:45.123456",
  "coefficients": [0.125, 0.234, ...],
  "intercept": 0.5234,
  "mse": 0.0234,
  "mae": 0.1234,
  "r2": 0.8567
}
```

**Cache Lifecycle**:
1. Model is always trained on initialization
2. Cache age is checked (TTL: 1 day)
3. If cache is fresh (< 1 day old): Load cached metrics, use trained model
4. If cache is stale or missing: Use freshly computed metrics, update cache
5. This ensures models stay fresh while avoiding redundant computation

**Cache Benefits**:
- Fast model initialization
- Persistent model metrics across sessions
- Auditability (timestamp shows when model was trained)
- Easy rollback if needed

---

## Data Processing Pipeline

### Data Ingestion

**Input Format**: Excel workbook with multiple sheets

**Sheet Processing**:
```python
all_sheets = pd.read_excel(excel_path, sheet_name=None)
# Returns: {sheet_name: DataFrame, ...}
```

**Data Flow**:
```
Excel File
    ↓
pd.read_excel() → Dictionary of DataFrames
    ↓
Filter by claim_id
    ↓
Convert to dictionary (row.to_dict())
    ↓
Nested dictionary structure
```

### Feature Engineering

**Fire Risk Features**:
```python
sprinkler_score = fire_internal.get("sprinkler_coverage_pct", 0) / 100
# Raw value: 0-100 (%) → Normalized: 0-1
```

**NatCat Features**:
```python
# Categorical mapping
flood_level_map = {"Low": 0.9, "Medium": 0.6, "High": 0.3}
flood_score = flood_level_map.get(external_flood_level, 0.6)

# Continuous normalization
wind_score = min(max_wind_gust_kmh / 150, 1.0)
# Assumes max reasonable wind is 150 km/h
```

**Operational Risk Features**:
```python
normalized_bi = min(bi_sum_insured / 1e8, 1)
# Assumes $100M = maximum BI threshold
```

### Data Quality Handling

**Missing Data Strategy**:
```python
# Sprinkler: defaults to 0 (worst case)
sprinkler_score = fire_internal.get("sprinkler_coverage_pct", 0) / 100

# Wind: defaults to 50 km/h (median)
if raw_wind is None or invalid:
    wind_val = 50.0
```

**Error Handling Pattern**:
```python
try:
    prediction = model.inference(features)
except Exception as e:
    logger.exception("Model inference failed, using fallback")
    # Fallback to linear weighted average
    score = weighted_average(features)
```

**Type Conversion Safety**:
```python
try:
    if raw_wind is not None and not isinstance(raw_wind, (str, bytes)):
        wind_val = float(raw_wind)
    else:
        wind_val = 50.0
except (ValueError, TypeError):
    logger.warning("Invalid wind data; using default 50.0")
    wind_val = 50.0
```

---

## External Data Integration

### ExternalRiskDataFetcher Architecture

**Class Purpose**: Centralized interface for external risk data sources

**Data Sources**:
1. **Geocoding**: Nominatim (OpenStreetMap)
2. **Weather**: Meteostat (3-year historical data)
3. **Seismic**: USGS Earthquake Feed (last 30 days, M≥4.5)
4. **Fire Brigade**: OSMnx + NetworkX (network routing)
5. **Flood**: Elevation-based proxy

**Implementation Notes** (partially commented in source):

```python
class ExternalRiskDataFetcher:
    def __init__(self, avg_fire_truck_speed_kmh=35):
        """
        Initialize external data fetcher.
        
        Args:
            avg_fire_truck_speed_kmh: Used to estimate response time
        """
        self.geolocator = Nominatim(user_agent="risk_engineering_crm")
        self.avg_speed = avg_fire_truck_speed_kmh
    
    def get_location(self, address: str) -> dict:
        """Geocode address to coordinates."""
        loc = self.geolocator.geocode(address)
        return {
            "address": address,
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "source": "OpenStreetMap",
            "confidence": "High"
        }
    
    def get_weather_risk(self, lat: float, lon: float) -> dict:
        """
        Fetch 3-year historical weather data.
        
        Returns:
            dict: max_wind_speed_kmh, max_wind_gust_kmh, source
        """
        point = Point(lat, lon)
        start = datetime.now() - timedelta(days=365*3)
        data = Daily(point, start, datetime.now()).fetch()
        
        wspd = data["wspd"].max()
        wpgt = data["wpgt"].max()
        
        return {
            "max_wind_speed_kmh": round(float(wspd), 1) if pd.notna(wspd) else None,
            "max_wind_gust_kmh": round(float(wpgt), 1) if pd.notna(wpgt) else None,
            "source": "Meteostat"
        }
    
    def get_earthquake_risk(self, lat: float, lon: float, 
                          radius_km: int = 300) -> dict:
        """
        Query USGS earthquake feed for seismic activity.
        
        Algorithm:
            1. Fetch USGS feed (M≥4.5, last 30 days)
            2. Filter events within radius_km using haversine distance
            3. Return max magnitude and event count
        """
        feed_url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_month.atom"
        feed = feedparser.parse(feed_url)
        
        magnitudes = []
        for entry in feed.entries:
            eq_lat, eq_lon = map(float, entry.georss_point.split())
            distance = self._haversine(lat, lon, eq_lat, eq_lon)
            
            if distance <= radius_km:
                mag = float(entry.title.split("M")[1].split(",")[0])
                magnitudes.append(mag)
        
        max_mag = max(magnitudes) if magnitudes else 0
        
        return {
            "events_last_30_days": len(magnitudes),
            "max_magnitude": max_mag,
            "seismic_risk_level": self._classify_seismic(max_mag),
            "source": "USGS Earthquake Feed"
        }
    
    def get_fire_brigade_risk(self, lat: float, lon: float, 
                             search_radius_m: int = 5000) -> dict:
        """
        Calculate fire brigade response time using network routing.
        
        Algorithm:
            1. Build road network graph around property using OSMnx
            2. Query fire stations within search radius using OpenStreetMap
            3. Calculate shortest path distance to nearest station
            4. Estimate response time = distance / avg_speed
        
        Returns:
            dict: nearest_station_distance_km, estimated_response_time_min
        """
        G = ox.graph_from_point((lat, lon), dist=search_radius_m)
        fire_stations = ox.features_from_point((lat, lon), 
                                               tags={"amenity": "fire_station"},
                                               dist=search_radius_m)
        
        if fire_stations.empty:
            return {"fire_station_found": False}
        
        site_node = ox.distance.nearest_nodes(G, lon, lat)
        distances = []
        
        for geom in fire_stations.geometry:
            fs_lat = geom.centroid.y
            fs_lon = geom.centroid.x
            fs_node = ox.distance.nearest_nodes(G, fs_lon, fs_lat)
            
            try:
                dist_m = nx.shortest_path_length(G, site_node, fs_node, 
                                                 weight="length")
                distances.append(dist_m)
            except nx.NetworkXNoPath:
                continue
        
        min_km = min(distances) / 1000 if distances else 10
        response_time = (min_km / self.avg_speed) * 60
        
        return {
            "fire_station_found": True,
            "nearest_station_distance_km": round(min_km, 2),
            "estimated_response_time_min": round(response_time, 1),
            "source": "OpenStreetMap (OSMnx)"
        }
    
    def get_all_external_risk_data(self, address: str, 
                                   lat: float, lon: float) -> dict:
        """
        Master wrapper calling all external data sources.
        
        Returns aggregated dict with all external risk factors.
        """
        return {
            "weather": self.get_weather_risk(lat, lon),
            "earthquake": self.get_earthquake_risk(lat, lon),
            "fire_brigade": self.get_fire_brigade_risk(lat, lon),
            "flood": self.get_flood_risk(elevation_m=50),
            "generated_at": datetime.now().isoformat()
        }
```

### API Failure Handling

**Graceful Degradation**:
```python
def get_weather_risk(self, lat, lon):
    try:
        # Attempt to fetch data
        data = Daily(point, start, end).fetch()
        if data.empty:
            return {"max_wind_speed_kmh": None, "source": "No data found"}
    except Exception as e:
        # Return error indicator without crashing
        return {"max_wind_speed_kmh": None, "source": "Error"}
```

**Fallback Computation**:
```python
# In risk computation
if raw_wind is not None:
    wind_score = min(float(raw_wind) / 150, 1.0)
else:
    wind_score = 50.0 / 150  # Safe default
```

---

## Frontend Architecture

### Streamlit Application Structure

**Main Application**: `frontend.py`

#### 1. **Imports & Initialization**
```python
import streamlit as st
from streamlit_option_menu import option_menu
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage

model = ChatOllama(model="llama3.2:latest", temperature=0.1)
st.set_page_config(layout='wide')
```

#### 2. **Custom Components**

**StreamlitUIHandler (Logging Bridge)**:
```python
class StreamlitUIHandler(logging.Handler):
    """
    Custom logging handler that displays logs in Streamlit UI.
    
    Features:
        - Creates animated status boxes (st.status) per module
        - Tracks timing for each module
        - Shows progress with emojis (⏳ processing, ✅ success, ❌ error)
        - Automatically updates headers with completion messages
    
    Usage:
        ui_handler = StreamlitUIHandler(st.container())
        logger.addHandler(ui_handler)
    """
    
    def __init__(self, container):
        super().__init__()
        self.container = container
        self.status_boxes = {}      # st.status objects
        self.start_times = {}       # Timing data
        self.success_messages = {}  # Success message text
    
    def emit(self, record):
        log_name = record.name
        log_msg = str(record.msg)
        
        # Create status box if first message from this logger
        if log_name not in self.status_boxes:
            with self.container:
                self.start_times[log_name] = time.time()
                self.status_boxes[log_name] = st.status(
                    f"⏳ **{log_name}**: Processing...", 
                    expanded=True
                )
        
        # Update status box based on log level
        status_box = self.status_boxes[log_name]
        
        if record.levelno >= logging.ERROR:
            status_box.error(f"❌ {log_msg}")
        elif record.levelno >= logging.WARNING:
            status_box.warning(f"⚠️ {log_msg}")
        else:
            status_box.write(f"⚙️ {log_msg}")
    
    def finalize_all(self):
        """Mark all status boxes as complete with timing."""
        for name, box in self.status_boxes.items():
            duration = time.time() - self.start_times.get(name, time.time())
            msg = self.success_messages.get(name, "Step completed")
            box.update(
                label=f"✅ **{name}**: {msg} ({duration:.1f}s)",
                state="complete",
                expanded=False
            )
```

#### 3. **Page Structure**

**Dashboard View**:
```python
if st.session_state.page == 'dashboard':
    st.title("Risk Engineering Portal")
    search_q = st.text_input("Asset Search", 
                            placeholder="Enter Client ID or Property Name...")
    
    # Filter property dataframe
    prop_df = all_sheets.get('property_details', pd.DataFrame())
    filtered_df = prop_df[
        prop_df['client_id'].astype(str).str.contains(search_q) | 
        prop_df['property_name'].str.contains(search_q)
    ]
    
    # Display clickable property list
    for _, row in filtered_df.iterrows():
        col1, col2, col3 = st.columns([1, 4, 1.5])
        col1.markdown(f"**{row['client_id']}**")
        col2.markdown(f"**{row['property_name']}**")
        if col3.button("Analyze Profile", key=f"btn_{row['client_id']}"):
            go_to_details(row['client_id'])
```

**Details View**:
```python
elif st.session_state.page == 'details':
    cid = st.session_state.selected_id
    prop = all_sheets['property_details'][
        all_sheets['property_details']['client_id'] == cid
    ].iloc[0]
    
    # Display property header
    st.markdown(f"<h1>{prop['property_name']} | ID: {cid}</h1>", 
               unsafe_allow_html=True)
    
    # Display tabs for different insights
    tab1, tab2, tab3, tab4 = st.tabs([
        "Infrastructure", "Protection", "Hazards", "Operational Info"
    ])
```

#### 4. **Report Generation**

**PDF Building**:
```python
def build_pdf(report_text: str) -> bytes:
    """
    Generate PDF from text content.
    
    Uses ReportLab to create professional PDF with:
    - Standard A4 page size
    - Consistent styling
    - Proper paragraph spacing
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    
    styles = getSampleStyleSheet()
    story = []
    
    for line in report_text.split("\n"):
        if line.strip():
            story.append(Paragraph(line, styles["Normal"]))
            story.append(Spacer(1, 8))  # 8pt spacing
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
```

**Email Generation**:
```python
def generate_and_send_email(claim_id, grade, score, inspection_type, claim_result):
    """
    Generate inspection request email using LLaMA 3.2 and send via SMTP.
    
    Prompt Engineering:
        - Specifies role: Senior Risk Engineer
        - Defines output structure (subject, body, sections)
        - Sets tone based on risk level
        - Requests data requirements (PDF format, no scanned images)
    
    SMTP Configuration:
        - Uses Gmail SMTP server
        - TLS encryption
        - App-specific password authentication
    """
    llm = ChatOllama(model="llama3.2", temperature=0.3)
    
    prompt = f"""
    As a Senior Risk Engineer at RiskIO, write a formal inspection request email.
    
    CLIENT DETAILS: {claim_result}
    
    PRELIMINARY RISK DATA:
    - Assigned Grade: {grade}
    - Risk Score: {score}
    - Required Inspection: {inspection_type}
    
    REQUIREMENTS:
    1. Professional subject line with Claim ID
    2. Context: Explain risk engine findings
    3. Data Requirement: Final report MUST be textual PDF
    4. Tone: {'Urgent' if inspection_type == 'Physical' else 'Formal'}
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    email_body = response.content
    
    # Send via SMTP
    msg = MIMEMultipart()
    msg['From'] = GMAIL
    msg['To'] = "surveyor@email.com"
    msg['Subject'] = f"{inspection_type} Inspection - {claim_id}"
    msg.attach(MIMEText(email_body, 'plain'))
    
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(GMAIL, G_PASS)
        server.send_message(msg)
```

#### 5. **Session State Management**

```python
# Initialize session state
if 'assessment_done' not in st.session_state:
    st.session_state.assessment_done = False

if 'claim_result' not in st.session_state:
    st.session_state.claim_result = None

if 'workflow_step' not in st.session_state:
    st.session_state.workflow_step = "START"  # START → ASSESSMENT → INSPECTION → PROFILING

if 'ai_cache' not in st.session_state:
    st.session_state.ai_cache = {
        'infra': {},        # Cache for infrastructure summaries
        'protection': {},   # Cache for protection summaries
        'hazards': {},      # Cache for hazard summaries
        'mgmt': {},         # Cache for management summaries
        'final_report': {}  # Cache for final reports
    }
```

#### 6. **Styling**

```python
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 10px; 
        border-radius: 5px; 
    }
    .section-header { 
        color: #1c2e4a; 
        font-weight: 700; 
        border-bottom: 2px solid #0056b3; 
    }
    .data-label { 
        color: #6c757d; 
        font-size: 0.75rem; 
        font-weight: 700; 
    }
    </style>
""", unsafe_allow_html=True)
```

---

## API & Integration Points

### RiskAssessmentEngine API

**Public Interface**:
```python
# Initialize
engine = RiskAssessmentEngine(excel_path="/path/to/data.xlsx")

# Assess claim
result = engine.assess_claim(claim_id="C001")

# Returns structure:
{
    "claim_id": "C001",
    "internal": {
        "property_details": {...},
        "building_construction": {...},
        "site_and_operations": {...},
        # ... all other sheets
    },
    "external": {
        "weather": {...},
        "earthquake": {...},
        "fire_brigade": {...},
        "flood": {...},
        "generated_at": "2026-01-29T14:30:45.123456"
    },
    "derived": {
        "individual_scores": {
            "fire_risk_score": 0.65,
            "natcat_score": 0.58,
            "operational_risk_score": 0.72
        },
        "overall_risk_score": 0.64,
        "risk_grade": "D"
    }
}
```

### ML Model APIs

**Fire Risk Inference**:
```python
from src.ml_algo.fire_risk.model.fire_risk_model import FIRE_RISK_Model

fire_model = FIRE_RISK_Model(degree=2)

prediction = fire_model.inference({
    "sprinkler_score": 0.8,
    "fire_brigade_score": 0.7,
    "detection_score": 1,
    "housekeeping_score": 0.6
})
# Returns: 0.412 (float between 0-1)
```

**NatCat Inference**:
```python
from src.ml_algo.natcat.model.nat_cat_model import NATCAT_Model

natcat_model = NATCAT_Model(degree=2)

prediction = natcat_model.inference({
    "flood_score": 0.6,
    "earthquake_score": 0.9,
    "wind_score": 0.45
})
# Returns: 0.583 (float between 0-1)
```

**Operational Risk Inference**:
```python
from src.ml_algo.operational_risk.model.operational_risk_model import OPERATIONAL_RISK_Model

op_model = OPERATIONAL_RISK_Model(degree=2)

prediction = op_model.inference({
    "single_site": 1,
    "critical_process": 1,
    "normalized_bi": 0.75,
    "loss_flag": 0
})
# Returns: 0.285 (float between 0-1)
```

### External Data API

```python
from src.External_data_gather import ExternalRiskDataFetcher

fetcher = ExternalRiskDataFetcher(avg_fire_truck_speed_kmh=35)

# Get all external data
data = fetcher.get_all_external_risk_data(
    address="123 Main St, Springfield, USA",
    lat=39.7817,
    lon=-89.6501
)

# Returns:
{
    "weather": {
        "max_wind_speed_kmh": 42.3,
        "max_wind_gust_kmh": 58.5,
        "source": "Meteostat"
    },
    "earthquake": {
        "events_last_30_days": 2,
        "max_magnitude": 3.8,
        "seismic_risk_level": "Low",
        "source": "USGS Earthquake Feed"
    },
    "fire_brigade": {
        "fire_station_found": True,
        "nearest_station_distance_km": 4.2,
        "estimated_response_time_min": 8.0,
        "source": "OpenStreetMap (OSMnx)"
    },
    "flood": {
        "flood_risk_level": "Low",
        "elevation_m": 50
    },
    "generated_at": "2026-01-29T14:30:45.123456"
}
```

### LLaMA 3.2 Integration

```python
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage

# Initialize model
model = ChatOllama(model="llama3.2:latest", temperature=0.1)

# Generate content
response = model.invoke([
    HumanMessage(content="Write a professional risk summary...")
])

email_body = response.content
```

**Temperature Settings**:
- `temperature=0.1`: For deterministic outputs (risk reports)
- `temperature=0.3`: For varied professional tone (emails)

---

## Development Workflow

### Setting Up Development Environment

**1. Clone & Navigate**:
```bash
cd Risk_Engineering
```

**2. Create Virtual Environment**:
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
source venv/bin/activate      # Unix/macOS
```

**3. Install Dependencies**:
```bash
pip install -r requirements.txt
```

**4. Configure Environment**:
```bash
# Create .env file
echo "FILE_PATH=path/to/your/data.xlsx" > .env
echo "GMAIL=your_email@gmail.com" >> .env
echo "G_PASS=your_app_password" >> .env
echo "SMTP_SERVER=smtp.gmail.com" >> .env
echo "SMTP_PORT=587" >> .env
```

**5. Test Installation**:
```bash
python -c "from src.gatheralldata import RiskAssessmentEngine; print('✅ Import successful')"
```

### Adding New Features

#### Example: Adding a New Risk Category

**Step 1: Create ML Model**
```python
# src/ml_algo/new_risk/model/new_risk_model.py
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class NEW_RISK_Model:
    def __init__(self, degree=2):
        # Load dataset, train, cache metrics
        pass
    
    def inference(self, user_input):
        # Predict new risk score
        pass
```

**Step 2: Add Dataset**
```
src/ml_algo/new_risk/data/
├── new_risk_dataset.csv     # Training data
└── new_risk_cache.json      # Cached metrics
```

**Step 3: Integrate into Engine**
```python
# In src/gatheralldata.py
from src.ml_algo.new_risk.model.new_risk_model import NEW_RISK_Model

new_risk_model = NEW_RISK_Model(degree=2)

# In compute_derived_metrics():
new_risk_user_input = { /* extracted features */ }
new_risk_score = new_risk_model.inference(new_risk_user_input)

# Update weighting:
# Old: 0.4*fire + 0.4*natcat + 0.2*operational
# New: 0.3*fire + 0.3*natcat + 0.2*operational + 0.2*new_risk
overall_risk_score = round(
    0.3*fire_risk_score + 0.3*natcat_score + 
    0.2*operational_risk_score + 0.2*new_risk_score, 2
)
```

**Step 4: Update Frontend**
```python
# In frontend.py, update score table display:
scores = derived.get("individual scores", {})
# Will automatically display new_risk_score from derived data
```

### Testing

**Unit Test Template**:
```python
# test.py
import sys
sys.path.append('.')

from src.gatheralldata import RiskAssessmentEngine
from src.ml_algo.fire_risk.model.fire_risk_model import FIRE_RISK_Model

def test_fire_model():
    model = FIRE_RISK_Model(degree=2)
    
    user_input = {
        "sprinkler_score": 0.8,
        "fire_brigade_score": 0.7,
        "detection_score": 1,
        "housekeeping_score": 0.6
    }
    
    score = model.inference(user_input)
    assert 0 <= score <= 1, f"Score out of range: {score}"
    print(f"✅ Fire Model Test Passed: {score}")

def test_engine():
    engine = RiskAssessmentEngine("data/data.xlsx")
    result = engine.assess_claim("C001")
    
    assert "derived" in result
    assert "overall_risk_score" in result["derived"]
    assert "risk_grade" in result["derived"]
    print(f"✅ Engine Test Passed: Grade {result['derived']['risk_grade']}")

if __name__ == "__main__":
    test_fire_model()
    test_engine()
```

**Run Tests**:
```bash
python test.py
```

---

## Troubleshooting & Debugging

### Common Issues

#### 1. **Module Import Errors**

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```python
# Add to sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
```

**Alternative in gatheralldata.py**:
```python
try:
    from src.ml_algo.fire_risk.model.fire_risk_model import FIRE_RISK_Model
except ImportError:
    # Fallback when running from src directory
    from ml_algo.fire_risk.model.fire_risk_model import FIRE_RISK_Model
```

#### 2. **Excel File Not Found**

**Problem**: `FileNotFoundError: [Errno 2] No such file or directory: 'data.xlsx'`

**Solution**:
```python
import os
from pathlib import Path

# Verify file exists
file_path = os.getenv("FILE_PATH", "data/data.xlsx")
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Data file not found at: {file_path}")

engine = RiskAssessmentEngine(file_path)
```

#### 3. **Missing Excel Columns**

**Problem**: `KeyError: 'client_id'`

**Solution**:
```python
# Check sheet columns
df = all_sheets['property_details']
if 'client_id' not in df.columns:
    raise ValueError(f"Missing 'client_id' column. Available: {df.columns.tolist()}")
```

#### 4. **External Data API Failures**

**Problem**: USGS API timeout, Meteostat returns empty data

**Solution**:
```python
import time
from requests.exceptions import Timeout

try:
    # Attempt with timeout
    data = Daily(point, start, end).fetch()
except Timeout:
    logger.warning("Meteostat timeout; using default values")
    data = None

if data is None or data.empty:
    # Use safe defaults
    wind_score = 50.0 / 150
```

#### 5. **LLaMA Model Not Available**

**Problem**: `Connection refused` when calling ChatOllama

**Solution**:
```bash
# Ensure Ollama is running
ollama serve

# In another terminal, pull the model
ollama pull llama3.2:latest
```

**Fallback in code**:
```python
try:
    model = ChatOllama(model="llama3.2:latest", temperature=0.1)
except Exception as e:
    logger.error(f"LLaMA unavailable: {e}")
    # Use dummy response for testing
    model = None
```

#### 6. **Memory Issues with Large Datasets**

**Problem**: `MemoryError` when loading large Excel files

**Solution**:
```python
# Load sheets selectively
sheets_needed = ['property_details', 'building_construction']
sheets = {}
for sheet in sheets_needed:
    sheets[sheet] = pd.read_excel(excel_path, sheet_name=sheet)

# Or use chunking for large sheets
for chunk in pd.read_excel(excel_path, sheet_name='large_data', chunksize=10000):
    process(chunk)
```

### Debugging Tools

**Enable Verbose Logging**:
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
```

**Check Cache Age**:
```python
import json
from datetime import datetime

with open('src/ml_algo/fire_risk/data/fire_risk_cache.json') as f:
    cache = json.load(f)
    
cache_time = datetime.fromisoformat(cache["timestamp"])
age_hours = (datetime.now() - cache_time).total_seconds() / 3600
print(f"Cache age: {age_hours:.1f} hours")
```

**Profile Performance**:
```python
import time

start = time.time()
result = engine.assess_claim("C001")
elapsed = time.time() - start

print(f"Assessment took {elapsed:.2f} seconds")
```

---

## Performance & Optimization

### Caching Strategy

**Current Caching Layers**:

1. **Model Cache** (JSON files)
   - TTL: 1 day
   - Contents: Model coefficients, metrics
   - Location: `ml_algo/*/data/*_cache.json`

2. **Streamlit Cache** (`@st.cache_data`)
   - Duration: Session-based
   - Contents: Excel DataFrames
   - Usage: `load_data(FILE_PATH)`

3. **Session State Cache**
   - Duration: Until browser refresh
   - Contents: AI summaries, assessment results
   - Location: `st.session_state.ai_cache`

**Optimization Opportunities**:

1. **Add Results Cache**:
```python
import sqlite3

# Cache assessment results
with sqlite3.connect('assessments.db') as conn:
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessments (
            claim_id TEXT PRIMARY KEY,
            result JSON,
            timestamp DATETIME
        )
    ''')
```

2. **Implement Async External Data Calls**:
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def get_all_external_data_async(address, lat, lon):
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor() as executor:
        weather = await loop.run_in_executor(
            executor, self.get_weather_risk, lat, lon
        )
        earthquake = await loop.run_in_executor(
            executor, self.get_earthquake_risk, lat, lon
        )
        # ... other calls in parallel
    
    return {
        "weather": weather,
        "earthquake": earthquake,
        # ...
    }
```

3. **Optimize Excel Reading**:
```python
# Instead of loading all sheets, load on-demand
def get_internal_data(self, claim_id):
    internal_data = {}
    
    for sheet_name in ['property_details', 'building_construction', ...]:
        try:
            df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
            row = df[df['client_id'] == claim_id]
            if not row.empty:
                internal_data[sheet_name] = row.iloc[0].to_dict()
        except Exception as e:
            logger.warning(f"Failed to load {sheet_name}: {e}")
    
    return internal_data
```

### Performance Benchmarks

**Typical Assessment Timeline**:
- Excel data loading: 100-500ms
- External data gathering: 2-5s (API dependent)
- ML model inference: 10-50ms (all 3 models)
- Report generation: 500-1000ms (LLaMA inference)
- **Total**: 3-7 seconds

**Optimization Targets**:
- External API calls (largest bottleneck)
- LLaMA inference (temperature affects speed/quality)
- Excel file size (consider CSV or Parquet for large datasets)

---

## Deployment Considerations

### Production Deployment

**Environment Setup**:
```bash
# Production .env
FILE_PATH=/secure/path/to/production_data.xlsx
GMAIL=alerts@company.com
G_PASS=$(aws secretsmanager get-secret-value --secret-id gmail-password)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# LLaMA Model
OLLAMA_BASE_URL=http://localhost:11434
```

**Docker Deployment** (Example):
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Database Considerations**:
- Current: File-based Excel
- Recommended for production: PostgreSQL with CSV imports
- Consider: S3 for remote data storage

### Security Best Practices

1. **Secrets Management**:
```python
from dotenv import load_dotenv
import os

load_dotenv()
GMAIL = os.getenv("GMAIL")
G_PASS = os.getenv("G_PASS")  # Use app-specific password, not actual password
```

2. **Input Validation**:
```python
def validate_claim_id(claim_id):
    if not isinstance(claim_id, str):
        raise ValueError("claim_id must be string")
    if len(claim_id) > 50:
        raise ValueError("claim_id too long")
    if not claim_id.replace('_', '').isalnum():
        raise ValueError("Invalid characters in claim_id")
    return claim_id
```

3. **Error Handling** (No sensitive data in logs):
```python
try:
    # Process
except Exception as e:
    logger.error("Assessment failed", exc_info=False)  # No stack trace in prod
    st.error("Assessment unavailable. Support: support@company.com")
```

---

## Maintenance & Monitoring

### Key Metrics to Monitor

1. **Model Performance**
   - R² score trending
   - Prediction distribution
   - Outlier analysis

2. **External API Reliability**
   - USGS feed availability
   - Meteostat success rate
   - OSMnx routing success

3. **System Health**
   - Cache hit rate
   - Assessment latency
   - Error rate

### Scheduled Tasks

```python
# Daily model retraining (via cron)
0 2 * * * python -c "from src.ml_algo.fire_risk.model.fire_risk_model import FIRE_RISK_Model; FIRE_RISK_Model(degree=2)"

# Weekly cache cleanup
0 3 * * 0 find /path/to/cache -type f -mtime +7 -delete

# Monthly data backup
0 4 1 * * rsync -av /path/to/data /backup/location
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Jan 2026 | Initial production release |

---

## Contributing

When contributing:

1. Follow existing code style (PEP 8)
2. Add logging for new features
3. Update cache handling if modifying ML models
4. Test with sample data before submission
5. Update README if adding new capabilities

---

## Support & Contact

- **Issues**: GitHub Issues or internal ticket system
- **Questions**: Contact development team
- **Production Emergencies**: Escalate to platform team

---

**Document Version**: 1.0.0  
**Last Updated**: January 2026  
**Maintained By**: Development Team
