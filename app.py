import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="SMS Spam Detection", page_icon="📩")
st.title("📩 SMS Spam Detection")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("spam_dense.keras")

model = load_model()

text = st.text_area("Enter SMS")

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter a message")
    else:
        # ✅ FIX: convert input to tf.string tensor
        pred = model.predict(tf.constant([text]))[0][0]

        if pred >= 0.5:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Ham Message")
