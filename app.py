import streamlit as st
import tensorflow as tf

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- Custom CSS ----------------
st.markdown("""
<style>

/* Full page background */
.stApp {
    background: linear-gradient(135deg, #667eea, #764ba2);
}

/* Main card */
.main-card {
    background: white;
    border-radius: 16px;
    padding: 40px;
    max-width: 900px;
    margin: 40px auto;
    box-shadow: 0 15px 40px rgba(0,0,0,0.15);
}

/* Title */
.title {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 30px;
}

/* Result cards */
.result {
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    font-size: 1.3rem;
    font-weight: 600;
    margin-top: 20px;
}

.spam {
    background: linear-gradient(135deg, #ff6b6b, #ee5a6f);
    color: white;
}

.ham {
    background: linear-gradient(135deg, #51cf66, #37b24d);
    color: white;
}

/* Confidence bar */
.bar {
    margin-top: 15px;
    height: 8px;
    background: rgba(255,255,255,0.3);
    border-radius: 4px;
    overflow: hidden;
}

.fill {
    height: 100%;
    background: white;
    transition: width 0.4s ease;
}

/* Button fix */
div.stButton > button {
    font-size: 1.1rem;
    padding: 0.6rem;
    border-radius: 10px;
}

/* Footer */
.footer {
    text-align: center;
    color: #999;
    margin-top: 30px;
    font-size: 0.9rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------- App UI ----------------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

st.markdown("<div class='title'>📩 SMS Spam Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered spam classification</div>", unsafe_allow_html=True)
st.divider()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("spam_dense.keras")

model = load_model()

# Input
col1, col2 = st.columns([3, 1])

with col1:
    text = st.text_area(
        "📝 Enter SMS",
        placeholder="Type or paste the message...",
        height=120
    )

with col2:
    st.markdown("### 📊 Stats")
    st.metric("Characters", len(text.strip()))
    st.metric("Words", len(text.split()))

st.divider()

# Button centered
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict = st.button("🔍 Analyze Message", use_container_width=True)

# Prediction
if predict:
    if not text.strip():
        st.warning("Please enter a message.")
    else:
        with st.spinner("Analyzing..."):
            pred = model.predict(tf.constant([text]), verbose=0)[0][0]
            confidence = pred*100 if pred >= 0.5 else (1-pred)*100

        if pred >= 0.5:
            st.markdown(f"""
            <div class="result spam">
                🚫 SPAM DETECTED
                <div class="bar"><div class="fill" style="width:{confidence}%"></div></div>
                Confidence: {confidence:.1f}%
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result ham">
                ✅ LEGITIMATE MESSAGE
                <div class="bar"><div class="fill" style="width:{confidence}%"></div></div>
                Confidence: {confidence:.1f}%
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🔍 Details")
        c1, c2 = st.columns(2)
        c1.info(f"Spam Score: {pred:.4f}")
        c2.info(f"Message Length: {len(text)} characters")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>🤖 Powered by TensorFlow · Built with Streamlit</div>", unsafe_allow_html=True)
