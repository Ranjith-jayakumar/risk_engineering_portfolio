import streamlit as st
import pandas as pd
from datetime import datetime
from langchain_ollama import ChatOllama
import json
from streamlit_option_menu import option_menu
import markdown
import logging
from langchain.messages import HumanMessage
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from PyPDF2 import PdfReader
# Initialize Model
from dotenv import load_dotenv
load_dotenv()

from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4


model = ChatOllama(model="llama3.2:latest", temperature=0.1)
st.set_page_config(layout='wide')
# --- ENGINE MOCK/IMPORT ---
try:
    from src.gatheralldata import RiskAssessmentEngine
except ImportError:
    class RiskAssessmentEngine:
        def __init__(self, path): pass
        def assess_claim(self, cid):
            return {
                "derived": {"risk_grade": "Moderate", "overall_risk_score": 0.65},
                "external": {
                    "earthquake": {"seismic_risk_level": "Low"},
                    "flood": {"flood_risk_level": "Medium"},
                    "fire_brigade": {"nearest_station_distance_km": 4.2, "estimated_response_time_min": 8}
                }
            }

import time
import logging
import streamlit as st

class StreamlitUIHandler(logging.Handler):
    def __init__(self, container):
        super().__init__()
        self.container = container
        self.status_boxes = {}      # Stores st.status objects
        self.start_times = {}       # Stores start time for each module
        self.success_messages = {}  # Stores the specific success log message

    def emit(self, record):
        log_name = record.name
        log_msg = str(record.msg)
        
        # 1. Start a new 'Slide Box' if it doesn't exist
        if log_name not in self.status_boxes:
            with self.container:
                self.start_times[log_name] = time.time()
                self.status_boxes[log_name] = st.status(f"⏳ **{log_name}**: Processing...", expanded=True)
                self.success_messages[log_name] = "Finished" # Default

        status_box = self.status_boxes[log_name]

        # 2. Logic to identify success messages
        success_keywords = ["successfully", "completed", "complete", "finished", "gathered"]
        
        if record.levelno >= logging.ERROR:
            status_box.error(f"❌ {log_msg}")
        elif record.levelno >= logging.WARNING:
            status_box.warning(f"⚠️ {log_msg}")
        else:
            if any(key in log_msg.lower() for key in success_keywords):
                self.success_messages[log_name] = log_msg
                status_box.write(f"✅ {log_msg}")
            else:
                status_box.write(f"⚙️ {log_msg}")

    def finalize_all(self):
        """Updates headers with specific success messages and duration."""
        for name, box in self.status_boxes.items():
            duration = time.time() - self.start_times.get(name, time.time())
            msg = self.success_messages.get(name, "Step completed")
            
            # Update header: ✅ InternalData: Data fetched successfully (1.2s)
            final_label = f"✅ **{name}**: {msg} ({duration:.1f}s)"
            box.update(label=final_label, state="complete", expanded=False)
# Set page configuration



###################
G_PASS=os.getenv("G_PASS")
GMAIL=os.getenv("GMAIL")
SMTP_SERVER=os.getenv("SMTP_SERVER")
SMTP_PORT=os.getenv("SMTP_PORT")

FILE_PATH = os.getenv("FILE_PATH")

# Place this at the start of your script
if "assessment_done" not in st.session_state:
    st.session_state.assessment_done = False
if "claim_result" not in st.session_state:
    st.session_state.claim_result = None
if "workflow_step" not in st.session_state:
    st.session_state.workflow_step = "START"

def build_pdf(report_text: str) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    styles = getSampleStyleSheet()
    story = []

    for line in report_text.split("\n"):
        if line.strip():
            story.append(Paragraph(line, styles["Normal"]))
            story.append(Spacer(1, 8))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def generate_and_send_email(claim_id, grade, score, inspection_type,claim_result):
    # 1. Generate Body with Llama 3.2
    llm = ChatOllama(model="llama3.2", temperature=0.3)
    prompt = f"""
    As a Senior Risk Engineer at RiskIO, write a formal inspection request email.

    CLIENT DETAILS:
    {claim_result}
    
    PRELIMINARY RISK DATA:
    - Assigned Grade: {grade}
    - Risk Score: {score}
    - Required Inspection: {inspection_type}

    INSTRUCTIONS FOR THE INSPECTOR:
    1. SUBJECT: Professional line with Claim ID and Inspection Type.
    2. CONTEXT: Inform the inspector that our Risk Engine identified this risk level and requires site verification.
    3. DATA REQUIREMENT: Explicitly state that the final risk report MUST be submitted as a textual PDF document (not scanned images) to allow for automated AI processing.
    4. CLOSING: Sign off with "Best Regards, RiskIO".
    5. FOOTER: Add the disclaimer: "This is an auto-generated mail. Do not reply."

    TONE: Professional, structured, and {'Urgent' if inspection_type == 'Physical' else 'Formal'}.
    """
    
    with st.spinner(f"Llama 3.2 is drafting your {inspection_type} request..."):
        response = llm.invoke([HumanMessage(content=prompt)])
        email_body = response.content
        subject = f"{inspection_type} Inspection Required - Claim {claim_id}"

    # 2. Send via SMTP
    try:
        msg = MIMEMultipart()
        msg['From'] = GMAIL
        msg['To'] = "abcd@gmail.com"
        msg['Subject'] = subject
        msg.attach(MIMEText(email_body, 'plain'))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(GMAIL, G_PASS)
            server.send_message(msg)
        
        st.success(f"✅ Email sent to surveyor for {inspection_type} inspection.")
        return True
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")
        return False


def file_processing_workflow():
    st.subheader("📤 Inspection Document Upload")
    st.info("Please upload the surveyor's notes to finalize the Risk Profile.")
    
    uploaded_file = st.file_uploader("Upload Inspection PDF", type=['pdf'])
    
    if uploaded_file is not None:
        try:
            # Extract text from PDF
            reader = PdfReader(uploaded_file)
            extracted_text = ""
            for page in reader.pages:
                extracted_text += page.extract_text()
            
            # Save extracted text to memory
            st.session_state.extracted_text = extracted_text
            st.success("✅ Document processed and text extracted.")
            
            # Button to move to the final stage
            if st.button("🔍 Process Risk Profiling"):
                st.session_state.workflow_step = "PROFILING"
                st.rerun()
                
        except Exception as e:
            st.error(f"Error reading PDF: {e}")


def generate_final_profile_report(inspection_notes):
    st.subheader("📋 Final Risk Profile Report")
    llm = ChatOllama(model="llama3.2", temperature=0.3)
    # Get original data from memory
    res = st.session_state.claim_result
    grade = res['derived']['risk_grade']
    
    # Create a simple report layout
    with st.expander("Preview Final Report", expanded=True):
        prompt = f"""
You are a Senior Insurance Risk Engineer and Property Underwriter.

Your task is to transform raw risk engine outputs into a
clear, professional, decision-ready Property Risk Profile
used by real-world insurance underwriters.

Write in formal underwriting language.
Avoid AI phrasing, marketing tone, or casual language.
Be factual, concise, and structured.

-----------------------
###PROPERTY RISK PROFILE
---------------------

Claim Reference:
- Claim ID: {st.session_state.claim_result.get('claim_id')}
- Assessment Date: {datetime.now().strftime('%Y-%m-%d')}
- Risk Grade: {grade}
- Overall Risk Score: {res['derived']['overall_risk_score']}

------------------------
1. Executive Risk Summary
------------------------
Provide a concise summary of the overall property risk condition.
Explain what the assigned risk grade means in underwriting terms
and whether the property is suitable for standard, conditional,
or restricted coverage.

------------------------
2. Property Risk Assessment
------------------------
Summarize the key structural, operational, and environmental
risk factors identified by the risk engine.
Focus on material exposures that could impact loss frequency
or severity.

------------------------
3. Fire & Hazard Exposure
------------------------
Analyze the fire risk characteristics of the property,
including construction considerations, ignition sources,
and exposure to surrounding hazards.
Reference the fire risk score where relevant.

------------------------
4. Risk Mitigation & Controls
------------------------
Evaluate existing safeguards, protections, and controls.
Indicate whether current measures are adequate,
partially adequate, or insufficient from an underwriting perspective.

------------------------
5. Inspection & Verification Notes
------------------------
Summarize the inspection findings below.
If information is incomplete, clearly state limitations
and recommend further verification if required.

Inspection Notes:
{inspection_notes[:1000]}

------------------------
6. Underwriting Considerations
------------------------
Highlight any concerns, assumptions, or dependencies
that underwriters should consider when pricing or approving coverage.

------------------------
7. Recommendation
------------------------
Provide a clear underwriting recommendation using one of the following:
- Accept at standard terms
- Accept with conditions
- Refer for further inspection
- Decline / Defer pending mitigation


Formatting rules:
- Plain text only
- No markdown
- No bullets or symbols
- Use clear section titles followed by paragraphs
- Suitable for direct PDF generation


Include disclaimers that  this content was generated by ai using client workflow.
Do NOT invent facts not present in the data.
Base conclusions strictly on the information provided.
"""
        response = llm.invoke([HumanMessage(content=prompt)]).content
        st.markdown(response)
    pdf_bytes = build_pdf(response)

    st.download_button(
        label="📥 Download Full PDF Report",
        data=pdf_bytes,
        file_name=f"Risk_Report_{st.session_state.claim_result.get('claim_id')}.pdf",
        mime="application/pdf"
    )


##############


# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 5px; border: 1px solid #e6e9ef; }
    .disclaimer { font-size: 0.75rem; color: #888; font-style: italic; margin-top: 15px; border-top: 1px solid #eee; padding-top: 5px; }
    .section-header { color: #1c2e4a; font-weight: 700; border-bottom: 2px solid #0056b3; padding-bottom: 5px; margin-bottom: 15px; text-transform: uppercase; font-size: 0.9rem; }
    .data-label { color: #6c757d; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
    .data-value { color: #212529; font-size: 0.95rem; font-weight: 500; margin-bottom: 8px; }
    .hazard-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #f0f2f6; }
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data(file_path):
    xls = pd.ExcelFile(file_path)
    return {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names}


def display_full_report(data):
    # --- LARGE HEADER SECTION ---
    # Assuming these are extracted from your data dict
    prop_name = data.get("internal", {}).get("property_details", {}).get("property_name", "N/A")
    cid = data.get("claim_id", "N/A")

    st.markdown(f"""
        <div style="text-align: left;">
            <h1 style="font-size: 50px; margin-bottom: 0px;">
                {prop_name} <span style="color: #666; font-weight: 300;"></span>
            </h1>
        </div>
        """, unsafe_allow_html=True)

    # --- MAIN TABS (Level 1) ---
    main_tab1, main_tab2, main_tab3 = st.tabs(["🏠 Internal Data", "🌍 External Data", "📈 Risk Scores"])

    with main_tab1:
        internal = data.get("internal", {})
        if internal:
            # Create sub-tabs dynamically based on the keys in the 'internal' dict
            sub_titles = [k.replace("_", " ").title() for k in internal.keys()]
            sub_tabs = st.tabs(sub_titles)
            
            for i, (section_name, values) in enumerate(internal.items()):
                with sub_tabs[i]:
                    st.subheader(f"{section_name.replace('_', ' ').title()}")
                    # Building the Table
                    table_md = "| Field | Value |\n| :--- | :--- |\n"
                    for key, val in values.items():
                        if isinstance(val, bool):
                            val = "✅ Yes" if val else "❌ No"
                        elif val is None:
                            val = "*N/A*"
                        
                        clean_key = key.replace("_", " ").capitalize()
                        table_md += f"| **{clean_key}** | {val} |\n"
                    st.markdown(table_md)

    with main_tab2:
        external = data.get("external", {})
        # Filter out keys that aren't dictionaries (like 'generated_at')
        sections = {k: v for k, v in external.items() if isinstance(v, dict)}
        
        if sections:
            sub_titles_ext = [k.replace("_", " ").title() for k in sections.keys()]
            sub_tabs_ext = st.tabs(sub_titles_ext)
            
            for i, (section_name, values) in enumerate(sections.items()):
                with sub_tabs_ext[i]:
                    st.subheader(f"External: {section_name.replace('_', ' ').title()}")
                    table_md = "| Field | Value |\n| :--- | :--- |\n"
                    for key, val in values.items():
                        if isinstance(val, bool):
                            val = "✅ Yes" if val else "❌ No"
                        elif val is None:
                            val = "*N/A*"
                        
                        clean_key = key.replace("_", " ").capitalize()
                        table_md += f"| **{clean_key}** | {val} |\n"
                    st.markdown(table_md)
        
        # Display the timestamp at the bottom of the External tab
        if "generated_at" in external:
            st.caption(f"Data generated at: {external['generated_at']}")

    with main_tab3:
        derived = data.get("derived", {})
        st.subheader("🎯 Risk Performance Summary")
        
        # Top level metrics
        c1, c2 = st.columns(2)
        
        # Logic for Grade Color
        grade = derived.get("risk_grade", "N/A")
        grade_color = "green" if grade in ["A"] else "orange" if grade in ["B","C","b","C"] else "red"
        
        c1.markdown(f"### Grade: <span style='color:{grade_color};'>{grade}</span>", unsafe_allow_html=True)
        c1.metric("Overall Score", f"{derived.get('overall_risk_score')}")

        st.divider()
        st.write("### Category Breakdown")
        
        scores = derived.get("individual scores", {})
        if scores:
            # Building the Score Table
            score_table = "| Risk Category | Score | Status |\n| :--- | :--- | :--- |\n"
            
            for category, value in scores.items():
                clean_cat = category.replace("_", " ").title()
                
                # Simple status indicator based on score (assuming 0-1 scale)
                if value >= 0.8:
                    status = "🟢 Excellent"
                elif value >= 0.5:
                    status = "🟡 Fair"
                else:
                    status = "🔴 Action Required"
                
                score_table += f"| **{clean_cat}** | {value} | {status} |\n"
            
            st.markdown(score_table)

try:
    all_sheets = load_data(FILE_PATH)
except Exception as e:
    st.error(f"System Error: Unable to access database at {FILE_PATH}")
    st.stop()



# --- STATE MANAGEMENT ---
if 'page' not in st.session_state: st.session_state.page = 'dashboard'
if 'selected_id' not in st.session_state: st.session_state.selected_id = None
if 'prelim_report' not in st.session_state: st.session_state.prelim_report = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []

# Persistent Cache for AI Summaries per Tab and Client ID
if 'ai_cache' not in st.session_state:
    st.session_state.ai_cache = {
        'infra': {},
        'protection': {},
        'hazards': {},
        'mgmt': {},
        'final_report': {}
    }

def go_to_details(cid):
    st.session_state.selected_id = cid
    st.session_state.page = 'details'
    st.session_state.prelim_report = None
    st.rerun()

def go_to_home():
    st.session_state.page = 'dashboard'
    st.session_state.selected_id = None
    st.rerun()

# --- HELPER: RISK BADGE ---
def get_risk_badge(level):
    val = str(level).upper()
    if val in ["HIGH", "TRUE", "ADVERSE", "PARTIAL","D","E"]:
        color, label = "#d9534f", "HIGH RISK"
    elif val in ["MEDIUM", "MODERATE","B","C"]:
        color, label = "#f0ad4e", "MODERATE"
    else:
        color, label = "#5cb85c", "LOW RISK"
    
    return f'<span style="color:white; background-color:{color}; padding:2px 10px; border-radius:12px; font-weight:bold; font-size:0.75rem;">{label}</span>'

# --- DASHBOARD PAGE ---
if st.session_state.page == 'dashboard':
    st.title("Risk Engineering Portal")
    search_q = st.text_input("Asset Search", placeholder="Enter Client ID or Property Name...").lower()
    
    prop_df = all_sheets.get('property_details', pd.DataFrame())
    
    if search_q:
        mask = (
            prop_df['client_id'].astype(str).str.lower().str.contains(search_q) | 
            prop_df['property_name'].str.lower().str.contains(search_q)
        )
        filtered_df = prop_df[mask.fillna(False)]
    else:
        filtered_df = prop_df

    with st.container(height=600):
        for _, row in filtered_df.iterrows():
            c1, c2, c3 = st.columns([1, 4, 1.5])
            c1.markdown(f"**{row['client_id']}**")
            c2.markdown(f"**{row['property_name']}**\n\n<span style='color: #6c757d; font-size: 0.85rem;'>{row['address']}, {row['city']}</span>", unsafe_allow_html=True)
            if c3.button("Analyze Profile", key=f"btn_nav_{row['client_id']}", use_container_width=True): 
                go_to_details(row['client_id'])
            st.markdown("---")

# --- DETAILS PAGE ---
elif st.session_state.page == 'details':
    cid = st.session_state.selected_id
    prop = all_sheets['property_details'][all_sheets['property_details']['client_id'] == cid].iloc[0]

    col_title,col_back = st.columns([5, 1])
    with col_back:
        if st.button("← Back", key="back_btn_unique"): 
            go_to_home()
    with col_title:
        # st.header(f"{prop['property_name']} | ID: {cid}")
        prop_name = prop['property_name']
        cid = cid

        # Custom CSS for a massive header
        st.markdown(f"""
            <div style="text-align: left;">
                <h1 style="font-size: 50px; margin-bottom: 0px;">
                    {prop_name} <span style="color: #666; font-weight: 300;">| ID: {cid}</span>
                </h1>
            </div>
            """, unsafe_allow_html=True)
        st.divider()

    # --- TOP SUMMARY OVERVIEW ---
    with st.container(border=True):
        sum_col1, sum_col2 = st.columns([4, 1])
        with sum_col1:
            acc = all_sheets.get('account_and_exposure', pd.DataFrame())
            k = acc[acc['client_id'] == cid].iloc[0] if not acc[acc['client_id'] == cid].empty else {}
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Occupancy", k.get('occupancy_type', 'N/A'))
            m2.metric("TIV", f"{k.get('currency','USD')} {k.get('total_insured_value',0):,.0f}")
            m3.metric("Limit", f"{k.get('max_location_limit',0):,.0f}")
            m4.metric("BI Sum Insured", f"{k.get('business_interruption_si',0):,.0f}")

    st.divider()
    st.header("Summary and insights",text_alignment="center")
    st.divider()
    selected_workflow = option_menu(
    menu_title=None, 
    options=["Insights", "Risk Summary"],
    # These are the Bootstrap Icon names
    icons=["lightbulb", "shield-check"], 
    default_index=0, 
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "#FF4B4B", "font-size": "16px"}, 
        "nav-link": {
            "font-size": "14px", 
            "text-align": "center", 
            "margin":"0px", 
            "--hover-color": "#eee"
        },
        "nav-link-selected": {
            "background-color": "#ffffff", 
            "color": "#FF4B4B", 
            "font-weight": "600", 
            "border": "1px solid #ff4b4b"
        },
    }
)
    # insights_tab,risk_summary_tab=st.tabs(["Insights", "Risk Summary"])
    if selected_workflow == "Insights":
                tab1, tab2, tab3, tab4= st.tabs(["Infrastructure", "Protection", "Hazards", "Operational Info"])

                with tab1:
                    st.markdown("<div class='section-header'>Facility & Infrastructure Details</div>", unsafe_allow_html=True)
                    s_row = all_sheets['site_and_operations'].query(f"client_id == '{cid}'").iloc[0]
                    b_row = all_sheets['building_construction'].query(f"client_id == '{cid}'").iloc[0]
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Construction:** {b_row['construction_type']}")
                        st.markdown(f"**Year Built:** {b_row['year_built']}")
                        st.markdown(f"**Floors:** {b_row['number_of_floors']} (Basement: {'Yes' if b_row['basement'] else 'No'})")
                        st.markdown(f"**Fire Resistance:** {b_row['fire_resistance_min']} mins")
                    with c2:
                        st.markdown(f"**Site Area:** {s_row['site_area_m2']:,} m²")
                        st.markdown(f"**Buildings:** {s_row['number_of_buildings']}")
                        st.markdown(f"**Occupancy Load:** {s_row['occupancy_load']} PAX")
                        st.markdown(f"**Operational Hours:** {s_row['operating_hours']}")
                    
                    st.divider()
                    st.markdown(f"**Criticality Rating:** {get_risk_badge(s_row['criticality'])}", unsafe_allow_html=True)
                    st.divider()
                    
                    # --- BUTTON TO TRIGGER AI SUMMARY ---
                    if st.button("✨ Generate Infrastructure Summary", key=f"btn_{cid}"):
                        with st.spinner("Generating..."):
                            gathered_data = f"""You are an intelligent risk engineer. Your job is to create a well-defined summary for the given Facility & Infrastructure Details:
                            - Construction: {b_row['construction_type']}
                            - Year Built: {b_row['year_built']}
                            - Floors: {b_row['number_of_floors']} (Basement: {'Yes' if b_row['basement'] else 'No'})
                            - Site Area: {s_row['site_area_m2']:,} m²
                            - Buildings: {s_row['number_of_buildings']}
                            """
                            # Store in session state cache
                            st.session_state.ai_cache['infra'][cid] = model.invoke(gathered_data).content

                    # --- DISPLAY SUMMARY IF IT EXISTS ---
                    if cid in st.session_state.ai_cache['infra']:
                        ai_response = st.session_state.ai_cache['infra'][cid]
                        st.markdown("**AI Summary:**")
                        st.markdown(
                            f"""<div style="height: 200px; overflow-y: auto; padding: 15px; border: 1px solid #e6e9ef; 
                            border-radius: 5px; background-color: #ffffff; font-size: 0.9rem; line-height: 1.5; color: #212529;">
                            {ai_response}</div>""", 
                            unsafe_allow_html=True
                        )
                        st.caption("Generated by Risk Engineering AI.")
                    else:
                        st.info("Click the button above to generate an automated site summary.")
                
                with tab2:
                    st.markdown("<div class='section-header'>Protection & Loss Prevention</div>", unsafe_allow_html=True)
                    f_row = all_sheets['fire_protection'].query(f"client_id == '{cid}'").iloc[0]
                    
                    pc1, pc2 = st.columns(2)
                    with pc1:
                        st.write(f"**Sprinkler Present:** {'✅ Yes' if f_row['sprinkler_present'] else '❌ No'}")
                        st.write(f"**Coverage:** {f_row['sprinkler_coverage_pct']}%")
                        st.write(f"**Fire Pump:** {'✅ Installed' if f_row['fire_pump'] else '❌ None'}")
                    with pc2:
                        st.write(f"**Water Supply:** {f_row['fire_water_supply']}")
                        st.write(f"**Detection System:** {f_row['detection_system']}")
                    
                    p_risk = "LOW" if f_row['sprinkler_present'] and f_row['sprinkler_coverage_pct'] > 80 else "HIGH"
                    st.markdown(f"**Protection Health:** {get_risk_badge(p_risk)}", unsafe_allow_html=True)
                    st.divider()
                    
                    # --- BUTTON TO TRIGGER PROTECTION ANALYSIS ---
                    if st.button("✨ Generate Protection Summary", key=f"prot_{cid}"):
                        with st.spinner("Analyzing Protection..."):
                            gathered_data = f"""You are a intelligent risk engineer your job is to create a well defined summary for the given Protection & Loss Prevention Details given below \n
                            - **Sprinkler Present:** {'Yes' if f_row['sprinkler_present'] else 'No'}.
                            - **Coverage:** {f_row['sprinkler_coverage_pct']}%.
                            - **Water Supply:** {f_row['fire_water_supply']}.
                            - **Detection System:** {f_row['detection_system']}.
                            - **Protection Health Rating:** {p_risk}
                            """
                            # Store in session state cache
                            st.session_state.ai_cache['protection'][cid] = model.invoke(gathered_data).content

                    # --- DISPLAY PROTECTION SUMMARY IF IT EXISTS ---
                    if cid in st.session_state.ai_cache['protection']:
                        ai_response = st.session_state.ai_cache['protection'][cid]
                        st.markdown("**AI Analysis Summary:**")
                        st.markdown(
                            f"""<div style="height: 200px; overflow-y: auto; padding: 15px; border: 1px solid #e6e9ef; 
                            border-radius: 5px; background-color: #ffffff; font-size: 0.9rem; line-height: 1.5; color: #212529;">
                            {ai_response}</div>""", 
                            unsafe_allow_html=True
                        )
                        st.caption("Detailed protection analysis generated by AI.")
                    else:
                        st.info("Click the button above to run the AI Protection & Loss Prevention analysis.")
                
                with tab3:
                    st.markdown("<div class='section-header'>Hazard Assessment</div>", unsafe_allow_html=True)
                    h_data = all_sheets['hazard_presence'].query(f"client_id == '{cid}'").iloc[0]
                    
                    haz_list = [
                        ("Flammable Materials", h_data['flammable_materials']),
                        ("High Voltage Systems", h_data['high_voltage']),
                        ("Thermal / High Temp", h_data['high_temperature']),
                        ("Pressure Systems", h_data['pressure_systems'])
                    ]
                    
                    for name, val in haz_list:
                        hc1, hc2 = st.columns([3, 1])
                        hc1.write(name)
                        hc2.markdown(get_risk_badge("HIGH" if val else "LOW"), unsafe_allow_html=True)
                        st.markdown("<hr style='margin:2px 0; border:0.1px solid #f0f2f6;'>", unsafe_allow_html=True)
                    
                    st.divider()

                    # --- BUTTON TO TRIGGER HAZARD EVALUATION ---
                    if st.button("✨ Generate Hazard Summary", key=f"haz_{cid}"):
                        with st.spinner("Evaluating Hazards..."):
                            # Formatting the list for a cleaner prompt
                            haz_summary_text = "\n".join([f"- {name}: {'Present (High Risk)' if val else 'Not Present (Low Risk)'}" for name, val in haz_list])
                            
                            gathered_data = f"""You are a intelligent risk engineer your job is to create a well defined summary for the given Hazard Assessment Details given below:
                            {haz_summary_text}
                            """
                            # Store in session state cache
                            st.session_state.ai_cache['hazards'][cid] = model.invoke(gathered_data).content

                    # --- DISPLAY HAZARD SUMMARY IF IT EXISTS ---
                    if cid in st.session_state.ai_cache['hazards']:
                        ai_response = st.session_state.ai_cache['hazards'][cid]
                        st.markdown("**AI Hazard Analysis:**")
                        st.markdown(
                            f"""<div style="height: 200px; overflow-y: auto; padding: 15px; border: 1px solid #e6e9ef; 
                            border-radius: 5px; background-color: #ffffff; font-size: 0.9rem; line-height: 1.5; color: #212529;">
                            {ai_response}</div>""", 
                            unsafe_allow_html=True
                        )
                        st.caption("Data source: Engineering Survey AI Evaluation")
                    else:
                        st.info("Click the button above to generate a hazard assessment summary.")
                
                with tab4:
                    st.markdown("<div class='section-header'>Risk Management & Loss History</div>", unsafe_allow_html=True)
                    rm = all_sheets['risk_management'].query(f"client_id == '{cid}'").iloc[0]
                    lh = all_sheets['loss_history'].query(f"client_id == '{cid}'").iloc[0]
                    
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.metric("Housekeeping", f"{rm['housekeeping_score']}/5")
                        st.write(f"**Maint Program:** {'✅ Active' if rm['preventive_maintenance'] else '❌ None'}")
                        st.write(f"**Emergency Plan:** {'✅ Documented' if rm['emergency_plan'] else '❌ Missing'}")
                        st.write(f"**Fire Drills:** {rm['fire_drills_per_year']} / year")
                    with rc2:
                        st.write(f"**5Y Loss History:** {'⚠️ Adverse' if lh['loss_last_5_years'] else '✅ Clean'}")
                        st.write(f"**Largest Loss:** ${lh['largest_loss_amount']:,}")
                        st.markdown(f"**Compliance:** {get_risk_badge(lh['regulatory_compliance'])}", unsafe_allow_html=True)

                    st.divider()
                    
                    # --- BUTTON TO TRIGGER RISK MGMT ANALYSIS ---
                    if st.button("✨ Generate Operational summary", key=f"mgmt_{cid}"):
                        with st.spinner("Reviewing Management Details..."):
                            gathered_data = f"""You are a intelligent risk engineer your job is to create a well defined summary for the given Risk Management & Loss History Details:
                            - Housekeeping Score: {rm['housekeeping_score']}/5
                            - Preventive Maintenance: {'Active' if rm['preventive_maintenance'] else 'None'}
                            - Emergency Plan: {'Documented' if rm['emergency_plan'] else 'Missing'}
                            - Fire Drills: {rm['fire_drills_per_year']} per year
                            - 5Y Loss History: {'Adverse events recorded' if lh['loss_last_5_years'] else 'Clean record'}
                            - Largest Recorded Loss: ${lh['largest_loss_amount']:,}
                            - Compliance Rating: {lh['regulatory_compliance']}
                            """
                            # Store in session state cache
                            st.session_state.ai_cache['mgmt'][cid] = model.invoke(gathered_data).content
                    
                    # --- DISPLAY SUMMARY IF IT EXISTS ---
                    if cid in st.session_state.ai_cache['mgmt']:
                        ai_response = st.session_state.ai_cache['mgmt'][cid]
                        st.markdown("**AI Management Summary:**")
                        st.markdown(
                            f"""<div style="height: 200px; overflow-y: auto; padding: 15px; border: 1px solid #e6e9ef; 
                            border-radius: 5px; background-color: #ffffff; font-size: 0.9rem; line-height: 1.5; color: #212529;">
                            {ai_response}</div>""", 
                            unsafe_allow_html=True
                        )
                        st.caption("This is an AI-generated assessment based on historical and operational data.")
                    else:
                        st.info("Click the button above to analyze risk management and loss history.")
            
    if selected_workflow == "Risk Summary":
        # --- 1. DEFINE THE CONTAINER FOR THE BUTTON ---
        # This allows us to "hide" the button or show information
                report_placeholder = st.container()

                # --- 2. LOGIC: IF REPORT DOES NOT EXIST IN CACHE, SHOW BUTTON ---
                if cid not in st.session_state.ai_cache['final_report']:
                    with report_placeholder:
                        st.info("Comprehensive analysis not yet performed for this client.")
                        
                        # The "Beautiful" primary button
                        if st.button(
                            "✨ Generate Risk Management summary", 
                            key=f"full_proc_{cid}"
                        ):
                            # The button disappears from the UI the moment this block starts
                            # because the logic moves into the spinner/generation phase.
                            
                            with st.spinner("⏳ Step 1: Processing Engineering Data..."):
                                engine = RiskAssessmentEngine(FILE_PATH)
                                # Run the engine
                                report_data = engine.assess_claim(cid)
                                st.session_state.prelim_report = report_data
                            
                            with st.spinner("✨ Step 2: AI Generating Comprehensive Report..."):
                                gathered_data = f"""
TASK:
Generate a professional insurance risk assessment report using the structured JSON data provided.

OUTPUT FORMAT (MANDATORY — FOLLOW EXACTLY):

# Risk Assessment Report

**Client:** <Client / Property Name>
**Location:** <City, State, Country>
**Report Date:** <Use today's date>

---

**Executive Summary**
A concise paragraph summarizing overall risk posture, major exposures, and mitigation effectiveness.

---

**Key Findings**
- **Risk Factors**
  - Bullet points listing the most material risk factors using plain language
  - Refer to conditions (e.g., flammable materials, high temperatures, asset concentration)
  - Do NOT show JSON keys or boolean values

- **Risk Mitigants**
  - Bullet points listing major controls and protections in place

---

**Hazard Presence**
Describe physical and operational hazards present at the site and their qualitative impact (Low / Moderate / High).

---

**Fire Protection**
Describe fire protection systems, response capability, and residual fire risk.

---

**Operational Risks**
Describe operational and business interruption exposures affecting loss severity and recovery.

---

**Loss History**
Summarize historical loss experience and its influence on current risk perception.

---

**Property Details**
- Address
- City
- State / Country
- Latitude / Longitude

---

**Location Details**
- Urban / Rural classification
- Industrial zone designation
- Surrounding exposure characteristics

---

**External Factors**
Summarize relevant external risk influences such as:
- Weather exposure
- Earthquake exposure
- Flood susceptibility

---

**Conclusion**
A concise underwriting-oriented conclusion stating overall risk level and insurability perspective.

---

## Explainability Summary

**Data Sources**
Explain how internal property data and external hazard data contributed to the assessment.

---

**Risk Factor Evaluation**
Explain how hazard presence, construction quality, protection systems, and operations influenced risk levels.

---

**Risk Interaction**
Explain how multiple risk factors combined or offset each other (e.g., hazards vs. protection).

---

**Risk Mitigation Effectiveness**
Explain why certain risks were moderated due to controls and management practices.

---

**Overall Risk Interpretation**
Explain how the final risk conclusion was reached without numeric formulas or internal scoring logic.

---

*Disclaimer: AI-generated risk summary for decision-support purposes only.*

STRICT RULES:
- Output ONLY the report content above
- DO NOT repeat or display JSON data
- DO NOT mention “AI”, “model”, or “risk engineer”
- DO NOT include chain-of-thought or internal reasoning
- Use professional insurance and underwriting language
- Use Markdown formatting exactly as shown
- Maintain consistency between risk conclusions and explainability

INPUT DATA:
{report_data}
"""

                                ai_content = model.invoke(gathered_data).content
                                
                                # Store in cache
                                st.session_state.ai_cache['final_report'][cid] = ai_content
                            
                            # Success notification and trigger UI refresh
                            st.toast("Report generated successfully!", icon="✅")
                            st.rerun()

                # --- 3. LOGIC: IF REPORT EXISTS, DISPLAY IT ---
                else:
                    # Get data from state
                    ai_response = st.session_state.ai_cache['final_report'][cid]
                    report = st.session_state.prelim_report
                    der = report.get('derived', {}) if report else {}

                    # Show the Grade Badge
                
                    st.markdown(f"### Final Risk Grade: {get_risk_badge(der.get('risk_grade', 'MODERATE'))}", unsafe_allow_html=True)
                    html_ai_response = markdown.markdown(ai_response, extensions=['extra'])

                    # 2. Inject the pre-rendered HTML into your styled div
                    st.markdown(
                        f"""
                        <div style="height: 550px; overflow-y: auto; padding: 20px; border: 1px solid #e6e9ef; 
                                    border-radius: 10px; background-color: #ffffff; font-size: 0.95rem; line-height: 1.6; color: #212529;
                                    box-shadow: inset 0 0 10px rgba(0,0,0,0.02);">
                            {html_ai_response}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                    
                    st.write("") 
                    
                    # Action Row (Download and Reset)
                    col_dl, col_rs = st.columns([3, 1])
                    with col_dl:
                        full_markdown_report = f"\n\n{ai_response}\n\n---\n*Disclaimer: AI-generated risk summary.*"
                        st.download_button(
                            label="📥 Download Full Risk Report (.md)", 
                            data=full_markdown_report, 
                            file_name=f"Risk_Report_{cid}.md", 
                            mime="text/markdown",
                            use_container_width=True
                        )
                    with col_rs:
                        if st.button("🔄 Reset", key=f"reset_final_{cid}", use_container_width=True):
                            st.session_state.prelim_report = None
                            if cid in st.session_state.ai_cache['final_report']:
                                del st.session_state.ai_cache['final_report'][cid]
                            st.rerun()
        
        # with right_col:

    st.divider()
    st.header("WORKFLOW",text_alignment="center")
    st.divider()

    selected_workflow = option_menu(
        menu_title=None, 
        options=["Risk Engineering Assessment", "Risk Assistant Chat", "Risk Engineering Data"],
        icons=["clipboard-data", "chat-text", "database-gear"], 
        menu_icon="cast", 
        default_index=0, 
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#FF4B4B", "font-size": "16px"}, # Primary color for icons
            "nav-link": {"font-size": "14px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#ffffff", "color": "#FF4B4B", "font-weight": "600", "border": "1px solid #ff4b4b"},
        }
    )

    if selected_workflow == "Risk Engineering Assessment":
            st.markdown("""
                <style>
                .workflow-card {
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 5px solid #FF4B4B;
                    margin-bottom: 20px;
                }
                </style>
            """, unsafe_allow_html=True)

            st.title("🛡️ Modern Risk Assessment Engine")
            st.caption("Real-time telemetry and commentary from the Risk Analysis core.")

            process_btn = st.button("Start Processing", type="secondary")

            # ===============================
            # RUN ENGINE (BUTTON ONLY SETS STATE)
            # ===============================
            if process_btn:
                commentary_area = st.container()
                ui_handler = StreamlitUIHandler(commentary_area)
                root_logger = logging.getLogger()
                root_logger.addHandler(ui_handler)
                root_logger.setLevel(logging.INFO)

                try:
                    engine = RiskAssessmentEngine(FILE_PATH)
                    claim_result = engine.assess_claim(cid)
                    ui_handler.finalize_all()

                    st.session_state.claim_result = claim_result
                    st.session_state.assessment_done = True

                except Exception as e:
                    st.error(f"Critical Error: {str(e)}")

                finally:
                    root_logger.removeHandler(ui_handler)

            # ===============================
            # DISPLAY RESULTS (OUTSIDE BUTTON)
            # ===============================
            if st.session_state.get("assessment_done"):

                res = st.session_state.claim_result
                derived = res.get("derived", {})

                grade = derived.get("risk_grade")
                score = derived.get("overall_risk_score")
                fire_score = (
                    derived
                    .get("individual scores", {})   # keeping your key unchanged
                    .get("fire_risk_score")
                )

                st.divider()
                st.subheader(f"Results for {cid}")

                c1, c2, c3 = st.columns(3)
                c1.metric("Risk Grade", grade)
                c2.metric("Overall Score", score)
                
                
                # ===============================
                # ACTION BUTTONS (NOW THEY WORK)
                # ===============================
                if grade == "A":
                    st.success("🌟 Grade A: High Quality Risk.")
                    if st.button("📄 Generate Risk Profile Report"):
                        st.session_state.workflow_step = "PROFILING"

                elif grade == "B":
                    st.warning("⚠️ Grade B: Remote Inspection Suggested.")
                    if st.button("📧 Trigger Remote Inspection Email"):
                        if generate_and_send_email(cid, grade, score, "Remote",st.session_state.claim_result):
                            st.session_state.workflow_step = "FILE_UPLOAD"

                else:  # Grade C or D
                    st.error(f"🚨 Grade {grade}: Physical Inspection Mandatory.")
                    if st.button("📧 Trigger Physical Inspection Email"):
                        if generate_and_send_email(cid, grade, score, "Physical",st.session_state.claim_result):
                            st.session_state.workflow_step = "FILE_UPLOAD"

            # ===============================
            # FOLLOW-UP WORKFLOWS
            # ===============================
            if st.session_state.get("workflow_step") == "FILE_UPLOAD":
                st.divider()
                file_processing_workflow()

            if st.session_state.get("workflow_step") == "PROFILING":
                st.divider()
                generate_final_profile_report(
                    st.session_state.get("extracted_text", "")
                )

    elif selected_workflow == "Risk Assistant Chat":
        st.markdown("<div class='section-header'>Interactive Risk Assistant</div>", unsafe_allow_html=True)
        
        # Create the scrolling chat window
        chat_box = st.container(height=450)
        with chat_box:
            for chat in st.session_state.chat_history:
                with st.chat_message(chat["role"]): 
                    st.write(chat["content"])

        # Handle Input
        if prompt := st.chat_input("Ask about " + prop['property_name']):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Build prompt context
            report_context = st.session_state.prelim_report if st.session_state.prelim_report else "No data available."
            ans = f"PRELIMINARY RISK ASSESSMENT: {report_context}. USER QUESTION: {prompt}"
            
            # Invoke AI Model
            ai_response_chat = model.invoke(ans).content
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response_chat})
            st.rerun()
        
        # Clear button with a unique key based on client ID
        if st.button("Clear History", key=f"clear_chat_{cid}"):
            st.session_state.chat_history = []
            st.rerun()

    elif selected_workflow == "Risk Engineering Data":
        if st.session_state.prelim_report:
            report_data = st.session_state.prelim_report
            # Execute your custom display function
            display_full_report(report_data)
        else:
            st.info("Detailed data will appear here after processing the risk report.")


