"""
Algerian Forest Fire — FWI Prediction UI
Run with:  streamlit run streamlit_app.py
"""

import requests
import streamlit as st

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
API_BASE_URL = "https://fwi-prediction-5ca6.onrender.com"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"

st.set_page_config(
    page_title="Forest Fire FWI Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# Custom CSS — dark forest + ember aesthetic
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

/* ── Root & background ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d1117;
    color: #e6e1d8;
    font-family: 'Syne', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(255, 90, 20, 0.12), transparent),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(30, 80, 40, 0.10), transparent);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Typography ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 800; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background-color: #111820; border-right: 1px solid #1e2730; }

/* ── Streamlit input labels ── */
label, .stSlider label, [data-testid="stWidgetLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #8a9bb0 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Number inputs ── */
[data-testid="stNumberInput"] input {
    background: #161d26 !important;
    border: 1px solid #2a3545 !important;
    border-radius: 6px !important;
    color: #e6e1d8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #ff5a14 !important;
    box-shadow: 0 0 0 2px rgba(255, 90, 20, 0.15) !important;
}

/* ── Slider track and thumb ── */
[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, #ff5a14, #ffac14) !important;
}

/* ── Predict button ── */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #c93a0a 0%, #ff5a14 50%, #ffac14 100%);
    color: #fff;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.06em;
    border: none;
    border-radius: 10px;
    padding: 0.85rem 2rem;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s;
    box-shadow: 0 4px 24px rgba(255, 90, 20, 0.3);
}
div.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
    box-shadow: 0 8px 32px rgba(255, 90, 20, 0.45);
}
div.stButton > button:active { transform: translateY(0px); }

/* ── Cards ── */
.fwi-card {
    background: #161d26;
    border: 1px solid #1e2d3d;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.fwi-card-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a6070;
    margin-bottom: 0.4rem;
}

/* ── Result box ── */
.result-low {
    background: linear-gradient(135deg, #0d2418, #0f2d1a);
    border: 1px solid #1d5c2a;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-medium {
    background: linear-gradient(135deg, #231a08, #2d2008);
    border: 1px solid #6b4c0a;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-high {
    background: linear-gradient(135deg, #200c06, #2d1008);
    border: 1px solid #7a2010;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-score {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3.4rem;
    line-height: 1;
    margin: 0.5rem 0;
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 0.4rem;
}
.result-risk {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    margin-top: 0.6rem;
}

/* ── Divider ── */
hr { border-color: #1e2730 !important; }

/* ── Info/error boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* ── Feature group headings ── */
.group-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #ff5a14;
    border-left: 2px solid #ff5a14;
    padding-left: 0.6rem;
    margin: 1.2rem 0 0.8rem;
}

/* ── Metric strip ── */
.metric-strip {
    display: flex;
    gap: 1rem;
    margin-bottom: 0.5rem;
}
.metric-chip {
    background: #1a2535;
    border: 1px solid #2a3a50;
    border-radius: 8px;
    padding: 0.35rem 0.75rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #6a8aaa;
}
.metric-chip span { color: #e6e1d8; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def get_risk_info(fwi: float) -> dict:
    if fwi < 5:
        return {
            "level": "Low Risk",
            "emoji": "🟢",
            "css_class": "result-low",
            "color": "#2ecc71",
            "description": "Conditions are relatively safe. Fire spread potential is minimal.",
            "advice": "Standard monitoring protocols apply.",
        }
    elif fwi < 15:
        return {
            "level": "Moderate Risk",
            "emoji": "🟡",
            "css_class": "result-medium",
            "color": "#f39c12",
            "description": "Elevated fire danger. Conditions could support fire spread.",
            "advice": "Increased vigilance and patrol recommended.",
        }
    else:
        return {
            "level": "High Risk",
            "emoji": "🔴",
            "css_class": "result-high",
            "color": "#e74c3c",
            "description": "Extreme fire weather conditions. High probability of rapid fire spread.",
            "advice": "⚠️ Activate emergency protocols immediately.",
        }


def call_predict_api(payload: dict) -> dict:
    response = requests.post(
        PREDICT_ENDPOINT,
        json=payload,
        timeout=10,
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()
    return response.json()


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div style="padding: 2.5rem 0 1.5rem;">
    <div style="font-family:'DM Mono',monospace; font-size:0.7rem; letter-spacing:0.18em;
                text-transform:uppercase; color:#ff5a14; margin-bottom:0.5rem;">
        🔥 Algeria Forest Fire Intelligence System
    </div>
    <h1 style="font-size:2.6rem; font-weight:800; margin:0; line-height:1.1; color:#f0ebe3;">
        Fire Weather Index<br>
        <span style="color:#ff5a14;">Prediction</span> Engine
    </h1>
    <p style="color:#6a7d8e; font-family:'DM Mono',monospace; font-size:0.82rem;
              margin-top:0.8rem; max-width:560px; line-height:1.6;">
        Input real-time weather observations to predict the FWI score using a
        Ridge Regression model trained on the Algerian Forest Fires dataset.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ──────────────────────────────────────────────
# API status check (sidebar)
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    if st.button("🔁 Check API Connection"):
        try:
            r = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                data = r.json()
                st.success("API is online ✅")
                st.json(data)
            else:
                st.error(f"API returned {r.status_code}")
        except Exception:
            st.error("Cannot reach API ❌")

    st.divider()
    st.markdown("""
    <div style='font-family:"DM Mono",monospace; font-size:0.72rem; color:#4a6070; line-height:1.8;'>
    <strong style='color:#8a9bb0'>API Endpoint</strong><br>
    POST /predict<br><br>
    <strong style='color:#8a9bb0'>Model</strong><br>
    Ridge Regression<br><br>
    <strong style='color:#8a9bb0'>Dataset</strong><br>
    Algerian Forest Fires<br><br>
    <strong style='color:#8a9bb0'>Risk Thresholds</strong><br>
    🟢 Low    — FWI &lt; 5<br>
    🟡 Medium — FWI &lt; 15<br>
    🔴 High   — FWI ≥ 15
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Input layout — 2 columns
# ──────────────────────────────────────────────
left_col, right_col = st.columns([1.05, 0.95], gap="large")

with left_col:
    st.markdown("### 📋 Weather Observations")

    st.markdown('<div class="group-tag">🌡 Atmospheric Conditions</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=0.0, max_value=60.0, value=29.0, step=0.1,
            help="Ambient air temperature in degrees Celsius",
        )
    with c2:
        rh = st.number_input(
            "RH — Relative Humidity (%)",
            min_value=0.0, max_value=100.0, value=57.0, step=1.0,
            help="Relative humidity percentage",
        )

    c3, c4 = st.columns(2)
    with c3:
        ws = st.number_input(
            "Ws — Wind Speed (km/h)",
            min_value=0.0, max_value=100.0, value=18.0, step=0.5,
            help="Wind speed in kilometres per hour",
        )
    with c4:
        rain = st.number_input(
            "Rain (mm)",
            min_value=0.0, max_value=50.0, value=0.0, step=0.1,
            help="Rainfall accumulation in millimetres",
        )

    st.markdown('<div class="group-tag">🌲 Fire Weather Codes</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        ffmc = st.number_input(
            "FFMC — Fine Fuel Moisture Code",
            min_value=0.0, max_value=101.0, value=65.7, step=0.1,
            help="Moisture content of fine fuels (litter, grass). Higher = drier = more ignitable.",
        )
    with c6:
        dmc = st.number_input(
            "DMC — Duff Moisture Code",
            min_value=0.0, max_value=500.0, value=3.4, step=0.1,
            help="Moisture content of loosely-compacted decomposing organic matter.",
        )

    c7, c8 = st.columns(2)
    with c7:
        dc = st.number_input(
            "DC — Drought Code",
            min_value=0.0, max_value=1000.0, value=7.6, step=0.5,
            help="Seasonal drought effects on deep organic layers. Slow-reacting index.",
        )
    with c8:
        isi = st.number_input(
            "ISI — Initial Spread Index",
            min_value=0.0, max_value=60.0, value=1.3, step=0.1,
            help="Expected fire spread rate. Combines FFMC and wind speed.",
        )

    st.markdown('<div class="group-tag">📊 Composite Indices</div>', unsafe_allow_html=True)
    bui = st.slider(
        "BUI — Build Up Index",
        min_value=0.0, max_value=300.0, value=3.4, step=0.1,
        help="Total amount of fuel available to a fire. Combines DMC and DC.",
    )

with right_col:
    st.markdown("### 🎯 Prediction")

    # ── Live input summary ──
    st.markdown("""
    <div class="fwi-card">
        <div class="fwi-card-header">Current Input Summary</div>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    s1.metric("Temp", f"{temperature}°C")
    s2.metric("RH", f"{rh}%")
    s3.metric("Wind", f"{ws} km/h")
    s4, s5, s6 = st.columns(3)
    s4.metric("FFMC", f"{ffmc}")
    s5.metric("DMC", f"{dmc}")
    s6.metric("ISI", f"{isi}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Predict button ──
    predict_clicked = st.button("🔥 Predict Fire Weather Index", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Result display ──
    if predict_clicked:
        payload = {
            "Temperature": temperature,
            "RH": rh,
            "Ws": ws,
            "Rain": rain,
            "FFMC": ffmc,
            "DMC": dmc,
            "DC": dc,
            "ISI": isi,
            "BUI": bui,
        }

        with st.spinner("Calling prediction API..."):
            try:
                result = call_predict_api(payload)

                # Support both response key formats
                fwi_score = result.get("prediction") or result.get("fwi_prediction")

                if fwi_score is None:
                    st.error("❌ Unexpected response format from API.")
                    st.json(result)
                else:
                    fwi_score = float(fwi_score)
                    risk = get_risk_info(fwi_score)

                    st.markdown(f"""
                    <div class="{risk['css_class']}">
                        <div class="result-label">Predicted FWI Score</div>
                        <div class="result-score" style="color:{risk['color']}">
                            {fwi_score:.2f}
                        </div>
                        <div class="result-risk" style="color:{risk['color']}">
                            {risk['emoji']} {risk['level']}
                        </div>
                        <hr style="border-color:rgba(255,255,255,0.08);margin:1rem 0">
                        <div style="font-family:'DM Mono',monospace;font-size:0.78rem;
                                    color:#9aa8b5;line-height:1.6;">
                            {risk['description']}<br><br>
                            <strong style="color:#c5cdd5">{risk['advice']}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # FWI gauge bar
                    st.markdown("<br>", unsafe_allow_html=True)
                    clamped = min(fwi_score / 40.0, 1.0)
                    st.markdown(f"""
                    <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                                letter-spacing:0.1em;text-transform:uppercase;
                                color:#4a6070;margin-bottom:0.4rem;">
                        FWI Gauge (0 → 40+)
                    </div>
                    <div style="background:#1a2535;border-radius:6px;height:10px;overflow:hidden;">
                        <div style="width:{clamped*100:.1f}%;height:100%;
                            background:linear-gradient(90deg,#2ecc71,#f39c12,#e74c3c);
                            border-radius:6px;transition:width 0.6s ease;"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;
                                font-family:'DM Mono',monospace;font-size:0.62rem;
                                color:#4a6070;margin-top:0.3rem;">
                        <span>0 — Low</span><span>15 — High</span><span>40+</span>
                    </div>
                    """, unsafe_allow_html=True)

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ **Cannot connect to API.**\n\n"
                    "Make sure FastAPI is running:\n"
                    "```\nuvicorn app.main:app --reload\n```"
                )
            except requests.exceptions.Timeout:
                st.error("⏱️ **Request timed out.** The API took too long to respond.")
            except requests.exceptions.HTTPError as e:
                st.error(f"🚫 **API Error {e.response.status_code}:** {e.response.text}")
            except Exception as e:
                st.error(f"⚠️ **Unexpected error:** {str(e)}")

    else:
        # Placeholder before prediction
        st.markdown("""
        <div style="border:1px dashed #1e2d3d;border-radius:14px;padding:3rem 2rem;
                    text-align:center;color:#3a5060;">
            <div style="font-size:2.5rem;margin-bottom:0.8rem;">🌲</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.78rem;
                        letter-spacing:0.08em;text-transform:uppercase;">
                Awaiting input prediction
            </div>
            <div style="font-size:0.75rem;margin-top:0.5rem;color:#2a4050;">
                Fill in weather parameters → click Predict
            </div>
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;font-family:'DM Mono',monospace;font-size:0.68rem;
            color:#2a3a4a;padding:0.5rem 0 1rem;letter-spacing:0.06em;">
    Algeria Forest Fire FWI Prediction · Ridge Regression Model ·
    Data: Algerian Forest Fires Dataset
</div>
""", unsafe_allow_html=True)