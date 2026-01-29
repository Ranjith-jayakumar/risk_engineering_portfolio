# Risk Engineering Platform

![Risk Engineering](https://img.shields.io/badge/Risk%20Assessment-Advanced%20ML-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Overview

**Risk Engineering** is a comprehensive, enterprise-grade risk assessment and management platform that leverages machine learning, external data sources, and advanced analytics to provide accurate property and operational risk evaluations. The system integrates internal claim data with real-time external environmental data to deliver actionable risk intelligence for insurance underwriting and claims management.

### Key Capabilities

- **Multi-Factor Risk Assessment**: Evaluates fire, natural catastrophe, and operational risks simultaneously
- **Real-Time External Data Integration**: Incorporates weather, seismic, flood, and emergency response data
- **AI-Powered Reporting**: Generates professional risk profiles using LLaMA 3.2 language model
- **Interactive Web Interface**: User-friendly Streamlit-based dashboard for risk analysis
- **Machine Learning Models**: Polynomial regression models for accurate risk scoring
- **Document Processing**: Automated PDF extraction and processing of surveyor reports
- **Email Integration**: Automated inspection request generation and distribution

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Frontend (Streamlit)                        │
│         Interactive Dashboard & Report Generation               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼──────┐  ┌────────▼─────────┐  ┌───▼──────────────┐
│ Risk Engine  │  │ External Data    │  │ AI Agent         │
│ Orchestrator │  │ Fetcher          │  │ (LLaMA 3.2)      │
└───────┬──────┘  └────────┬─────────┘  └───┬──────────────┘
        │                  │                 │
        └──────────────────┼─────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼──────┐  ┌────────▼─────────┐  ┌───▼──────────────┐
│ Fire Risk    │  │ NatCat Risk      │  │ Operational Risk │
│ ML Model     │  │ ML Model         │  │ ML Model         │
└──────────────┘  └──────────────────┘  └──────────────────┘
```

---

## 🔧 Tech Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Frontend Framework** | Streamlit | Latest | Interactive web interface |
| **Language** | Python | 3.8+ | Core application logic |
| **ML/AI Framework** | scikit-learn | Latest | Machine learning models |
| **LLM** | LLaMA 3.2 (via Ollama) | Latest | Natural language processing |
| **Data Processing** | pandas, NumPy | Latest | Data manipulation & analysis |
| **Geospatial** | geopy, OSMnx, NetworkX | Latest | Location-based risk analysis |
| **Weather Data** | meteostat | Latest | Historical weather patterns |
| **LLM Framework** | LangChain | Latest | LLM orchestration & chains |
| **PDF Processing** | PyPDF2, ReportLab | Latest | Document handling |
| **Email** | smtplib | Built-in | SMTP email integration |

### Data & Infrastructure

- **Data Source**: Excel workbooks (XLSX format)
- **Cache System**: JSON-based model caching
- **Database**: File-based (Excel) with in-memory processing
- **External APIs**: USGS Earthquake Feed, Meteostat, OpenStreetMap

---

## 📦 Project Components

### 1. **Frontend Module** (`frontend.py`)
The main Streamlit application providing:
- Dashboard for asset search and navigation
- Detailed property analysis views
- Risk assessment visualization
- Interactive data exploration
- PDF report generation and download
- Email integration for inspection requests

**Key Features:**
- Multi-tab interface for internal/external data
- Real-time risk scoring metrics
- AI-powered summary generation
- Workflow management (assessment → inspection → profiling)
- Caching for AI summaries per client

### 2. **Risk Assessment Engine** (`src/gatheralldata.py`)
Core orchestration engine that:
- Aggregates internal claim data from Excel sheets
- Fetches external risk data
- Computes derived risk metrics
- Integrates ML model predictions
- Generates comprehensive risk assessments

**Main Classes:**
- `RiskAssessmentEngine`: Orchestrates the complete assessment workflow

### 3. **External Data Fetcher** (`src/External_data_gather.py`)
Collects real-time external risk factors:
- **Geocoding**: Location intelligence using Nominatim
- **Weather Risk**: Historical wind data from Meteostat
- **Seismic Risk**: Earthquake proximity analysis via USGS feed
- **Fire Brigade Response**: Network routing to nearest stations using OSMnx
- **Flood Risk**: Elevation-based flood hazard assessment

### 4. **Machine Learning Models** (`src/ml_algo/`)

#### Fire Risk Model (`fire_risk/`)
- **Features**: Sprinkler coverage, fire brigade proximity, detection systems, housekeeping
- **Algorithm**: Polynomial Regression (degree 2)
- **Output**: Fire risk score (0-1 scale)
- **Performance**: Cached metrics (MSE, MAE, R²)

#### Natural Catastrophe Model (`natcat/`)
- **Features**: Flood risk, earthquake severity, wind speed
- **Algorithm**: Polynomial Regression (degree 2)
- **Output**: NatCat risk score (0-1 scale)
- **Covers**: Climate and seismic exposures

#### Operational Risk Model (`operational_risk/`)
- **Features**: Single-site criticality, critical processes, business interruption, loss history
- **Algorithm**: Polynomial Regression (degree 2)
- **Output**: Operational risk score (0-1 scale)
- **Focus**: Business continuity impact

### 5. **AI Agents Module** (`AI_agents/`)
Supports advanced AI-powered workflows:
- Risk assessment orchestration
- Intelligent tool selection
- Multi-step reasoning
- Extensible agent patterns

### 6. **Data Assets** (`data/`)
Sample data for testing:
- `sample.json`: JSON format test data
- `sample.txt`: Text format examples

---

## 🎯 Risk Scoring Methodology

### Overall Risk Calculation
```
Overall Risk Score = 0.4 × Fire Risk + 0.4 × NatCat Risk + 0.2 × Operational Risk
```

### Risk Grade Assignment
| Score Range | Grade | Interpretation |
|-------------|-------|-----------------|
| 0.85 - 1.0 | A | Excellent (Lowest Risk) |
| 0.70 - 0.84 | B | Good |
| 0.65 - 0.69 | C | Fair |
| 0.00 - 0.64 | D | Action Required (Highest Risk) |

### Risk Factors

**Fire Risk Components:**
- Sprinkler system coverage and effectiveness
- Fire brigade proximity and response time
- Automatic detection systems
- Housekeeping and maintenance standards

**NatCat Risk Components:**
- Flood exposure level
- Seismic/earthquake risk
- Wind/storm exposure

**Operational Risk Components:**
- Single-site dependency
- Critical process reliance
- Business interruption exposure
- Loss history indicators

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Local Ollama installation (for LLaMA 3.2 model)
- SMTP credentials (for email functionality)

### Installation

1. **Clone the repository**
```bash
cd Risk_Engineering
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate  # Unix/macOS
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
Create a `.env` file in the root directory:
```env
FILE_PATH=path/to/your/data.xlsx
GMAIL=your_email@gmail.com
G_PASS=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

5. **Launch the application**
```bash
streamlit run frontend.py
```

The application will open in your default browser at `http://localhost:8501`

---

## 📊 Input Data Requirements

### Excel Workbook Format
The application expects an Excel file with the following sheets:

| Sheet | Purpose | Key Columns |
|-------|---------|------------|
| `property_details` | Property information | `client_id`, `property_name`, `address`, `city`, `latitude`, `longitude` |
| `building_construction` | Construction details | `construction_type`, `year_built`, `number_of_floors`, `basement`, `fire_resistance_min` |
| `site_and_operations` | Operational data | `site_area_m2`, `number_of_buildings`, `occupancy_load`, `criticality` |
| `account_and_exposure` | Coverage & exposure | `occupancy_type`, `total_insured_value`, `max_location_limit`, `business_interruption_si` |
| `fire_protection` | Fire safety measures | `sprinkler_coverage_pct`, `detection_system` |
| `risk_management` | Risk controls | `housekeeping_score` |
| `loss_history` | Historical loss data | `loss_last_5_years` |

---

## 🔄 Workflow

### Standard Assessment Workflow

1. **Dashboard Navigation**
   - Search for claims by Client ID or Property Name
   - Click "Analyze Profile" to initiate assessment

2. **Risk Assessment**
   - System fetches internal data from Excel
   - External data is gathered (weather, seismic, fire brigade)
   - ML models compute risk scores
   - Preliminary risk grade is assigned

3. **Inspection Request** (if high risk detected)
   - AI generates professional inspection request
   - Email is sent to surveyor with instructions
   - Surveyor uploads inspection report as PDF

4. **Final Risk Profile**
   - Inspection notes are processed
   - AI generates comprehensive underwriting report
   - PDF report is generated and available for download

---

## 📈 Output & Reports

### Assessment Output Includes:

**Structured JSON Report:**
```json
{
  "claim_id": "C001",
  "internal": { /* All internal data sheets */ },
  "external": { /* Weather, seismic, fire brigade data */ },
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

**Professional PDF Report:**
- Executive risk summary
- Detailed risk assessments by category
- Mitigation controls evaluation
- Inspection verification notes
- Underwriting recommendations

---

## 🔐 Security & Compliance

- **Email Security**: SMTP with TLS encryption
- **Data Privacy**: File-based storage with access controls
- **Model Integrity**: Cached model versioning with timestamps
- **Error Handling**: Comprehensive logging and fallback mechanisms

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🤝 Support & Contribution

For issues, questions, or contributions:
- Report bugs via GitHub Issues
- Submit pull requests for enhancements
- Contact the development team

---

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [LangChain Documentation](https://python.langchain.com)
- [scikit-learn ML Guide](https://scikit-learn.org)
- [OpenStreetMap/OSMnx Guide](https://osmnx.readthedocs.io)

---

**Last Updated**: January 2026  
**Version**: 1.0.0  
**Status**: Production Ready
