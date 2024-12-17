import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from skimage.transform import resize
from skimage.io import imread
import os
import time

# ---- Page Configuration ----
st.set_page_config(
    page_title="Musculoskeletal Abnormality Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Load the model ----
@st.cache_resource
def load_trained_model():
    model_path = "my_model.h5"  # Change this if your model is in a different format or path
    model = load_model(model_path)
    return model

model = load_trained_model()

# ---- Image preprocessing function ----
def preprocess_image(img_path, target_size=(224, 224)):
    """
    Reads an image, resizes it to the target size, crops to the center,
    and normalizes pixel values.
    """
    try:
        img = imread(img_path)
        img_resized = resize(img, (300, 300, 3))
        h, w, _ = img_resized.shape

        startx = w // 2 - (target_size[0] // 2)
        starty = h // 2 - (target_size[1] // 2)
        img_cropped = img_resized[starty:starty + target_size[1], startx:startx + target_size[0]]

        img_normalized = img_cropped / 255.0
        return img_normalized
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# ---- Class labels ----
CLASS_NAMES = ["Positive Abnormality", "Negative Abnormality"]

# ---- Example images ----
EXAMPLE_IMAGES = {
    "Example 1": "example_images/image1.png",
    "Example 2": "example_images/image2.png",
    "Example 3": "example_images/image3.png",
}

# ---- Main Streamlit App ----
def main():
    st.title(":hospital: Musculoskeletal Abnormality Detector")
    st.markdown("**This app detects abnormalities in X-ray images using a deep learning model.**")

    # Layout for better organization
    col1, col2 = st.columns([1, 2])

    # Sidebar for file upload and example selection
    with col1:
        st.subheader("Upload or Select an Image")
        uploaded_file = st.file_uploader(
            "Upload an X-ray Image", type=["png", "jpg", "jpeg"], help="Supported formats: PNG, JPG, JPEG"
        )
        st.markdown("---")
        st.subheader("Or choose an example image:")
        example_choice = st.selectbox("Choose an Example", list(EXAMPLE_IMAGES.keys()), index=0)

    # Image processing and prediction
    with col2:
        st.subheader("Image Preview")

        # Process uploaded file or example image
        if uploaded_file is not None:
            img_path = "temp_image.png"
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_to_process = img_path
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        else:
            image_to_process = EXAMPLE_IMAGES.get(example_choice)
            if image_to_process and os.path.exists(image_to_process):
                st.image(image_to_process, caption=f"Selected Example: {example_choice}", use_column_width=True)
            else:
                st.error(f"Error: The selected image '{example_choice}' is not available. Please check the file path.")

        # Processing steps
        if st.button("Analyze Image", key="analyze_button"):
            with st.spinner("Processing image... Please wait!"):
                time.sleep(1)  # Simulating loading time

                # Preprocess the image
                preprocessed_image = preprocess_image(image_to_process)

                if preprocessed_image is not None:
                    img_batch = np.expand_dims(preprocessed_image, axis=0)
                    predictions = model.predict(img_batch)
                    predicted_class = np.argmax(predictions, axis=1)[0]
                    confidence = predictions[0][predicted_class]

                    # Display results
                    st.success("### Prediction Result")
                    st.write(f"**Class:** {CLASS_NAMES[predicted_class]}")
                    st.progress(float(confidence))  # Confidence as progress bar
                    st.write(f"**Confidence:** {confidence:.2%}")
                else:
                    st.error("Failed to preprocess the image.")

    # Footer
    st.markdown("---")
    st.markdown("**Developed with :heart: by Dishant Chouhan,karnika deveradi and Kashish patidar**")

if __name__ == "__main__":
    main()
