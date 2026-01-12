import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    .stContainer {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 40px;
        margin: 20px auto;
        max-width: 900px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    .title-text {
        text-align: center;
        font-size: 2.8em;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .subtitle-text {
        text-align: center;
        color: #666;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1.3em;
        margin-top: 20px;
        animation: slideIn 0.5s ease-in-out;
    }
    .spam-result {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
    }
    .ham-result {
        background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
        color: white;
    }
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .confidence-bar {
        margin-top: 15px;
        height: 8px;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.3);
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    </style>
""", unsafe_allow_html=True)

# Main container
st.markdown('<div class="stContainer">', unsafe_allow_html=True)

st.markdown('<div class="title-text">📩 SMS Spam Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Advanced AI-powered spam classification with high accuracy</div>',
    unsafe_allow_html=True
)

st.divider()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("spam_dense.keras")

model = load_model()

# Input section with styling
col1, col2 = st.columns([3, 1], gap="medium")

with col1:
    text = st.text_area(
        "📝 Enter your SMS message",
        placeholder="Paste or type the message you want to analyze...",
        height=120,
        label_visibility="visible"
    )

with col2:
    st.markdown("### Message Stats")
    char_count = len(text.strip())
    word_count = len(text.split())
    st.metric("Characters", char_count)
    st.metric("Words", word_count)

st.divider()

# Prediction button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button(
        "🔍 Analyze Message",
        use_container_width=True,
        type="primary"
    )

if predict_btn:
    if not text.strip():
        st.warning("⚠️ Please enter a message to analyze")
    else:
        with st.spinner("Analyzing message..."):
            # Get prediction
            pred = model.predict(tf.constant([text]), verbose=0)[0][0]
            confidence = pred * 100 if pred >= 0.5 else (1 - pred) * 100

        if pred >= 0.5:
            st.markdown(
                f'''
                <div class="result-box spam-result">
                    🚫 SPAM DETECTED
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence}%"></div>
                    </div>
                    Confidence: {confidence:.1f}%
                </div>
                ''',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'''
                <div class="result-box ham-result">
                    ✅ LEGITIMATE MESSAGE
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence}%"></div>
                    </div>
                    Confidence: {confidence:.1f}%
                </div>
                ''',
                unsafe_allow_html=True
            )

        # Additional insights
        st.markdown("### 📊 Analysis Details")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Spam Score:** {pred:.4f}")
        with col2:
            st.info(f"**Message Length:** {len(text)} characters")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    ---
    <div style='text-align: center; color: #999; font-size: 0.9em; margin-top: 30px;'>
    🤖 Powered by TensorFlow | Built with Streamlit
    </div>
""", unsafe_allow_html=True)
