import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="I‑CONI Dashboard", layout="centered")

st.title("🦉 I‑CONI: Integrated Conservation Intelligence")
st.markdown("An AI-powered system for NDVI, Species Classification, and Poaching Risk Prediction.")

option = st.sidebar.selectbox("Choose Module", ["NDVI Monitor", "Wildlife Classifier", "Poaching Risk Predictor"])

# NDVI MONITOR
if option == "NDVI Monitor":
    import rasterio

    st.header("🛰️ NDVI Habitat Health Monitoring")
    uploaded_file = st.file_uploader("Upload a satellite image (.tif)", type=["tif"])

    if uploaded_file is not None:
        st.success("File uploaded successfully!")

        with rasterio.open(uploaded_file) as src:
            bands = src.read()

        if bands.shape[0] < 5:
            st.error("This image needs at least 5 bands (Band 4 - Red, Band 5 - NIR).")
        else:
            red = bands[3].astype('float32')  # Band 4
            nir = bands[4].astype('float32')  # Band 5

            ndvi = (nir - red) / (nir + red + 1e-6)
            ndvi = np.clip(ndvi, -1, 1)

            # Show NDVI heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            ndvi_plot = ax.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
            ax.set_title("NDVI Heatmap")
            plt.colorbar(ndvi_plot, ax=ax)
            st.pyplot(fig)

            # Show metrics
            st.metric("Mean NDVI", round(np.mean(ndvi), 3))
            st.metric("Min NDVI", round(np.min(ndvi), 3))
            st.metric("Max NDVI", round(np.max(ndvi), 3))

# WILDLIFE CLASSIFIER
elif option == "Wildlife Classifier":
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image

    st.header("🐾 Wildlife Species Classifier")
    uploaded_img = st.file_uploader("Upload an animal image", type=["jpg", "jpeg", "png"])

    if uploaded_img is not None:
        img = Image.open(uploaded_img).convert('RGB')
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Preprocess
        img = img.resize((150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Load model
        model_path = "model/wildlife_cnn.h5"
        model = load_model(model_path)

        # Define class names
        class_names = ['deer', 'elephant', 'lion', 'tiger']  # Update as per your model

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        st.success(f"Predicted Species: **{predicted_class}**")

# POACHING RISK PREDICTOR
elif option == "Poaching Risk Predictor":
    import pickle
    import pandas as pd

    st.header("🚨 Poaching Risk Predictor")

    # Input fields
    ndvi = st.number_input("NDVI Score (-1 to 1)", min_value=-1.0, max_value=1.0, step=0.01)
    density = st.number_input("Wildlife Density (animals per sq km)", min_value=0)
    patrols = st.number_input("Patrol Frequency (visits/week)", min_value=0)
    poaching = st.number_input("Poaching Incidents in Last 3 Months", min_value=0)

    if st.button("Predict Poaching Risk"):
        try:
            # Load model
            with open("poaching_risk_model.pkl", "rb") as f:
                model = pickle.load(f)

            # Create DataFrame with correct column names
            input_data = pd.DataFrame([{
                'ndvi': ndvi,
                'density': density,
                'patrols': patrols,
                'poaching': poaching
            }])

            # Predict
            prediction = model.predict(input_data)[0]

            # Decode and display result
            label_map = {0: "🟢 LOW", 1: "🟠 MEDIUM", 2: "🔴 HIGH"}
            st.success(f"Predicted Poaching Risk: {label_map.get(prediction, 'Unknown')}")

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

