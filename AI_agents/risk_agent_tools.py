# import sys
# import os
# import streamlit as st
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
# from gatheralldata import RiskAssessmentEngine
# import json

# engine = RiskAssessmentEngine(r"C:\Users\Ranjith\Desktop\Risk_Engineering\data\data.xlsx")
# claim_result = engine.assess_claim("C001")

# print(json.dumps(claim_result, indent=4))

import sys
import os
import json
import logging
import streamlit as st


# Setup path to import your src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from gatheralldata import RiskAssessmentEngine

# --- Custom Handler for Modern UI ---
class StreamlitUIHandler(logging.Handler):
    def __init__(self, container):
        super().__init__()
        self.container = container
        self.status_boxes = {} # Stores the "Slide Boxes" (st.status)

    def emit(self, record):
        log_name = record.name
        log_msg = record.msg
        
        # Check if we already have a 'Slide Box' for this module
        if log_name not in self.status_boxes:
            with self.container:
                # This creates the Header (Slide Box)
                self.status_boxes[log_name] = st.status(f"**{log_name}**", expanded=True)
        
        # Add the log message as a Sub-header with a spinner-like icon
        status_box = self.status_boxes[log_name]
        if record.levelno >= logging.ERROR:
            status_box.error(f"❌ {log_msg}")
        elif record.levelno >= logging.WARNING:
            status_box.warning(f"⚠️ {log_msg}")
        else:
            # Modern commentary style
            status_box.write(f"⚙️ {log_msg}")

# --- Streamlit UI Layout ---
st.set_page_config(page_title="Risk Assessment Engine", layout="wide")

st.title("🛡️ Modern Risk Assessment Engine")
st.caption("Real-time telemetry and commentary from the Risk Analysis core.")

# Sidebar Controls
with st.sidebar:
    st.header("Input Parameters")
    excel_path = st.text_input("Excel Path", r"C:\Users\Ranjith\Desktop\Risk_Engineering\data\data.xlsx")
    claim_id = st.text_input("Claim ID", "C001")
    process_btn = st.button("Start Processing", type="primary", use_container_width=True)

if process_btn:
    # Area for the animated commentary
    commentary_area = st.container()
    
    # Setup Logging Linkage
    ui_handler = StreamlitUIHandler(commentary_area)
    root_logger = logging.getLogger()
    root_logger.addHandler(ui_handler)
    root_logger.setLevel(logging.INFO)

    try:
        # Initialize Engine
        engine = RiskAssessmentEngine(excel_path)
        
        # Execute Assessment
        with st.spinner("Analyzing Risks..."):
            claim_result = engine.assess_claim(claim_id)
        
        # Close all 'Slide Boxes' (mark as complete)
        for box in ui_handler.status_boxes.values():
            box.update(label="Step Completed", state="complete", expanded=False)

        st.divider()
        
        # Visualize Results
        st.subheader(f"Results for {claim_id}")
        col1, col2, col3 = st.columns(3)
        
        derived = claim_result.get("derived", {})
        scores = derived.get("individual scores", {})
        
        col1.metric("Risk Grade", derived.get("risk_grade"))
        col2.metric("Overall Score", derived.get("overall_risk_score"))
        col3.metric("Fire Risk", scores.get("fire_risk_score"))

        with st.expander("View Raw Data Report"):
            st.json(claim_result)

    except Exception as e:
        st.error(f"Critical Error: {str(e)}")
    
    finally:
        # Clean up handler to prevent memory leaks/duplicate logs
        root_logger.removeHandler(ui_handler)

else:
    st.info("👈 Click 'Start Processing' in the sidebar to begin the engine.")





