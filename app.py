import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub

st.set_page_config(page_title="SMS Spam Detection", page_icon="📩")

st.title("📩 SMS Spam Detection")
st.write("Enter an SMS message to check whether it is Spam or Ham.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "spam_use_model.keras",
        custom_objects={"KerasLayer": hub.KerasLayer}
    )

model = load_model()

message = st.text_area("Enter SMS text")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([message])[0][0]
        if prediction >= 0.5:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Ham Message")
