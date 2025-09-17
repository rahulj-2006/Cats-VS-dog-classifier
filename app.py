import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
# Corrected the typo in the line below (mobilenet_v2)
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Animal Classifier",
    page_icon="🐾",
    layout="centered"
)

# --- Model Loading ---
# Cache the model so it's only loaded once
@st.cache_resource
def load_model():
    """Loads the pre-trained MobileNetV2 model."""
    model = MobileNetV2(weights="imagenet")
    return model

model = load_model()

# --- Prediction Function ---
def make_prediction(image, model):
    """
    Takes an image and a model, preprocesses the image,
    and returns the decoded predictions.
    """
    # Resize and preprocess for the model
    # Use Image.Resampling.LANCZOS for modern Pillow versions
    resized_img = image.resize((224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(resized_img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array)

    # Make prediction
    predictions = model.predict(processed_img)
    decoded = decode_predictions(predictions, top=3)[0]
    return decoded

# --- UI Components ---

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=100)
st.sidebar.title("🐾 DOG vs CAT Classifier")
st.sidebar.markdown("""
Upload an image of a pet or animal to see if it's a **Cat**, **Dog**, or something else. 
This app uses the **MobileNetV2** model pre-trained on ImageNet.
""")

# Main Page Title
st.title("🐕‍🦺 Cats vs Dogs Classifier")
st.caption("🔍 Upload an image to see what the AI thinks it is!")

# File Uploader
uploaded_file = st.file_uploader("📂 Choose a JPG or PNG image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Please upload an image file to get a prediction.")
else:
    try:
        # Open and display the image
        image = Image.open(uploaded_file).convert("RGB")
        
        # Use columns for a better layout
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.image(image, caption="📷 Uploaded Image", use_column_width=True)

        # Show a spinner while making the prediction
        with st.spinner('🧠 Analyzing the image...'):
            decoded_predictions = make_prediction(image, model)

            top_label = decoded_predictions[0][1].lower()
            confidence = decoded_predictions[0][2]

            # Custom classification logic
            if any(word in top_label for word in ["cat", "kitten", "siamese", "tabby", "persian"]):
                label = "Cat 🐱"
            elif any(word in top_label for word in ["dog", "puppy", "retriever", "shepherd", "terrier", "beagle"]):
                label = "Dog 🐶"
            else:
                label = "Other 🐾"

            # Display results in the second column
            with col2:
                st.metric("🎯 Top Prediction", value=label)
                st.write(f"**Best Match:** {decoded_predictions[0][1].replace('_', ' ').title()}")
                st.write(f"**Confidence:** {confidence:.2%}")

                with st.expander("📊 See Detailed Predictions"):
                    for i, (id, name, prob) in enumerate(decoded_predictions):
                        st.write(f"{i+1}. **{name.replace('_', ' ').title()}**: {prob:.2%}")

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")

