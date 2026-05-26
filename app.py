import streamlit as st
import numpy as np
import onnxruntime as ort

from PIL import Image

session = ort.InferenceSession(
    "road_damage_model.onnx"
)

input_name = session.get_inputs()[0].name

classes = {
    0: "Pothole",
    1: "Crack",
    2: "Manhole"
}

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚧"
)

st.title("🚧 AI-Based Road Damage Detection")

uploaded_file = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = img.resize((128,128))

    img_array = np.array(img).astype(np.float32)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = session.run(
        None,
        {input_name: img_array}
    )

    prediction = prediction[0]

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.success(
        f"Prediction: {classes[predicted_class]}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )
