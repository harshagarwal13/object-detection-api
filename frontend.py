import streamlit as st
import requests
from PIL import Image

# Backend URL
BACKEND_URL = "http://ai-backend:8005/detect/"

st.title("Object Detection App")
st.write("Upload an image to detect objects using Yolo")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Detecting objects...")

    with st.spinner("Processing..."):
        response = requests.post(
            BACKEND_URL,
            files={"file": uploaded_file.getvalue()}
        )

    if response.status_code == 200:
        detections = response.json().get("detections", [])
        st.write("### Detection Results:")
        for detection in detections:
            st.write(f"- **{detection['label']}** with confidence **{detection['score']:.2f}**")
    else:
        st.error("Error in backend processing!")
