import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from PIL import Image

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚧",
    layout="centered"
)

model = load_model("road_damage_model.keras")

classes = {
    0: "Pothole",
    1: "Crack",
    2: "Manhole"
}

st.title("🚧 AI-Based Road Damage Detection System")

st.subheader(
    "Smart City Infrastructure Monitoring using CNN"
)

st.markdown("---")

st.header("📌 About the Project")

st.write("""
Road monitoring is essential for maintaining safe transportation infrastructure.
Damaged roads can increase accidents, traffic congestion, and vehicle damage.

This project uses Convolutional Neural Networks (CNNs) to automatically analyze
road images and identify different types of road damage.

CNNs are powerful deep learning models used in computer vision applications
because they can automatically learn image features such as edges, cracks,
textures, and potholes.

### Industry Applications
- Smart City Monitoring
- Autonomous Vehicles
- Highway Maintenance Systems
- Municipal Infrastructure Management
- AI-Powered Road Inspection
""")

st.markdown("---")

st.header("📤 Upload Road Image")

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.markdown("---")

    st.header("🖼 Uploaded Image Preview")

    img = Image.open(uploaded_file)

    st.image(
        img,
        caption="Uploaded Road Image",
        use_container_width=True
    )

    resized_img = img.resize((128,128))

    img_array = image.img_to_array(resized_img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    predicted_label = classes[predicted_class]

    if predicted_label == "Pothole":
        severity = "High"

    elif predicted_label == "Crack":
        severity = "Medium"

    else:
        severity = "Low"

    st.markdown("---")

    st.header("🔍 Prediction Results")

    st.success(
        f"Prediction: {predicted_label} Detected"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    st.warning(
        f"Severity Level: {severity}"
    )

    st.markdown("---")

    st.header("📊 Confidence Visualization")

    probabilities = prediction[0] * 100

    fig, ax = plt.subplots(figsize=(6,4))

    ax.bar(
        classes.values(),
        probabilities
    )

    ax.set_ylabel("Confidence (%)")

    ax.set_xlabel("Classes")

    ax.set_title("Class Confidence Scores")

    st.pyplot(fig)

    st.markdown("---")

    st.header("🛠 Recommendations")

    if predicted_label == "Pothole":

        st.error("""
Immediate maintenance recommended.

High-risk road condition detected.
Potential danger to vehicles and public safety.
""")

    elif predicted_label == "Crack":

        st.warning("""
Road inspection recommended.

Cracks may expand over time and
lead to severe road deterioration.
""")

    else:

        st.success("""
Routine monitoring recommended.

Current road condition appears relatively stable.
""")