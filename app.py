import streamlit as st
import numpy as np
import tensorflow as tf

from PIL import Image

interpreter = tf.lite.Interpreter(
    model_path="road_damage_model.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = {
    0: "Pothole",
    1: "Crack",
    2: "Manhole"
}

st.title("Road Damage Detection")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","png","jpeg"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(img)

    img = img.resize((128,128))

    img_array = np.array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array.astype(np.float32),
        axis=0
    )

    interpreter.set_tensor(
        input_details[0]['index'],
        img_array
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details[0]['index']
    )

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.success(
        f"Prediction: {classes[predicted_class]}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )
