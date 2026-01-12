import streamlit as st
import tensorflow as tf

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== THEME TOGGLE ==================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# ================== CSS (GLASSMORPHISM) ==================
def glass_css(theme):
    if theme == "dark":
        bg = "linear-gradient(135deg,#667eea,#764ba2)"
        card = "rgba(255,255,255,0.15)"
        text = "white"
        subtext = "#e0e0e0"
    else:
        bg = "linear-gradient(135deg,#e0e7ff,#fdf2f8)"
        card = "rgba(255,255,255,0.6)"
        text = "#111"
        subtext = "#444"

    return f"""
    <style>
    .stApp {{
        background: {bg};
    }}

    .glass {{
        backdrop-filter: blur(14px);
        background: {card};
        border-radius: 20px;
        padding: 30px;
        max-width: 900px;
        margin: 30px auto;
        box-shadow: 0 15px 40px rgba(0,0,0,0.25);
        color: {text};
    }}

    h1 {{
        text-align: center;
        color: {text};
        font-weight: 700;
    }}

    .subtitle {{
        text-align: center;
        color: {subtext};
        margin-bottom: 25px;
    }}

    div.stButton > button {{
        background: linear-gradient(135deg,#4facfe,#00f2fe);
        color: white;
        font-size: 1.1rem;
        padding: 0.7rem;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }}

    .result {{
        margin: 20px auto;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        color: white;
        max-width: 800px;
    }}

    .spam {{
        background: linear-gradient(135deg,#ff6b6b,#ee5a6f);
    }}

    .ham {{
        background: linear-gradient(135deg,#51cf66,#37b24d);
    }}

    .bar {{
        height: 10px;
        background: rgba(255,255,255,0.3);
        border-radius: 6px;
        overflow: hidden;
        margin-top: 10px;
    }}

    .fill {{
        height: 100%;
        background: white;
    }}

    .footer {{
        text-align: center;
        margin-top: 30px;
        color: {subtext};
        font-size: 0.9rem;
    }}

    @media (max-width: 768px) {{
        .glass {{
            padding: 20px;
            margin: 15px;
        }}
    }}
    </style>
    """

st.markdown(glass_css(st.session_state.theme), unsafe_allow_html=True)

# ================== THEME TOGGLE BUTTON ==================
col1, col2, col3 = st.columns([1,1,1])
with col3:
    st.button(
        "🌙 Dark / ☀️ Light",
        on_click=toggle_theme,
        use_container_width=True
    )

# ================== MAIN CARD ==================
st.markdown("<div class='glass'>", unsafe_allow_html=True)

st.markdown("<h1>📩 SMS Spam Detection</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered spam classification</div>", unsafe_allow_html=True)
st.divider()

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("spam_dense.keras")

model = load_model()

# ================== INPUT SECTION ==================
col1, col2 = st.columns([3,1])

with col1:
    text = st.text_area(
        "📝 Enter SMS",
        placeholder="Type or paste the message here...",
        height=120
    )

with col2:
    st.markdown("### 📊 Stats")
    st.metric("Characters", len(text.strip()))
    st.metric("Words", len(text.split()))

st.divider()

# ================== BUTTON ==================
col1, col2, col3 = st.columns([1,2,1])
with col2:
    analyze = st.button("🔍 Analyze Message", use_container_width=True)

# ================== PREDICTION ==================
if analyze:
    if not text.strip():
        st.warning("Please enter a message.")
    else:
        with st.spinner("Analyzing..."):
            pred = model.predict(tf.constant([text]), verbose=0)[0][0]
            confidence = pred*100 if pred >= 0.5 else (1-pred)*100

        if pred >= 0.5:
            st.markdown(f"""
            <div class="result spam">
                <h2>🚫 SPAM DETECTED</h2>
                <div class="bar">
                    <div class="fill" style="width:{confidence}%"></div>
                </div>
                <p>Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result ham">
                <h2>✅ LEGITIMATE MESSAGE</h2>
                <div class="bar">
                    <div class="fill" style="width:{confidence}%"></div>
                </div>
                <p>Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🔍 Details")
        c1, c2 = st.columns(2)
        c1.info(f"Spam Score: {pred:.4f}")
        c2.info(f"Message Length: {len(text)} characters")

st.markdown("</div>", unsafe_allow_html=True)

# ================== FOOTER ==================
st.markdown("<div class='footer'>🤖 Powered by TensorFlow · Built with Streamlit</div>", unsafe_allow_html=True)
