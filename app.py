
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Alzheimer's MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

class_names = [
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_alzheimer_model.keras")

model = load_model()


def predict_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index] * 100

    return (
        class_names[predicted_index],
        confidence,
        predictions
    )


st.title("🧠 Alzheimer's MRI Classification")

st.write(
    "Upload a brain MRI image to obtain a prediction from the trained AI model."
)

st.warning(
    "For educational and research purposes only. "
    "This application is not a medical diagnostic tool."
)

uploaded_file = st.file_uploader(
    "Upload MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded MRI",
        use_container_width=True
    )

    if st.button("🔍 Predict"):

        with st.spinner("Analyzing image..."):

            predicted_class, confidence, predictions = predict_image(image)

        st.success(f"Prediction: {predicted_class}")

        st.metric(
            "Model Probability",
            f"{confidence:.2f}%"
        )

        st.subheader("Class Probabilities")

        for class_name, probability in zip(
            class_names,
            predictions
        ):
            st.write(
                f"**{class_name}**: {probability * 100:.2f}%"
            )

            st.progress(float(probability))
