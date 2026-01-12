import streamlit as st
import tensorflow as tf

st.title("📩 SMS Spam Detection")

model = tf.keras.models.load_model("spam_dense.keras")

text = st.text_area("Enter SMS")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Enter a message")
    else:
        pred = model.predict([text])[0][0]
        st.success("🚫 Spam" if pred >= 0.5 else "✅ Ham")
