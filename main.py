import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from streamlit_folium import folium_static
import folium


st.set_page_config(page_title="Plant Identifier & Mapper", layout="wide")


model = ResNet50(weights='imagenet')


st.title("🌿 Smart Plant Identifier & Reforestation Mapper")
st.markdown("""
Upload or capture an image of a plant to identify it using AI. 
Then, mark its location on the map for reforestation tracking.
""")


with st.sidebar:
    st.header("📸 Image Input")
    use_camera = st.radio("Choose image input method:", ["Upload Image", "Capture via Camera"])

    
    def capture_image():
        cam = cv2.VideoCapture(0)
        while True:
            ret, frame = cam.read()
            if not ret:
                st.error("Failed to capture image")
                break
            cv2.imshow("Press 'c' to capture, 'q' to quit", frame)
            key = cv2.waitKey(1)
            if key == ord('c'):
                cv2.imwrite('captured_image.jpg', frame)
                break
            elif key == ord('q'):
                st.info("Capture cancelled")
                break
        cam.release()
        cv2.destroyAllWindows()

    uploaded_file = None
    if use_camera == "Capture via Camera":
        if st.button("📷 Capture Image"):
            capture_image()
            uploaded_file = 'captured_image.jpg'
    else:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📷 Uploaded Image")
        img = image.load_img(uploaded_file, target_size=(224, 224))
        st.image(img, caption='Input Image', use_column_width=True)

    with col2:
        st.subheader("🔍 Predictions")
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        decoded_preds = decode_predictions(preds, top=3)[0]

        labels = [label for (_, label, _) in decoded_preds]
        scores = [score for (_, _, score) in decoded_preds]

        for i, (_, label, score) in enumerate(decoded_preds):
            st.markdown(f"**{i+1}. {label}** — {score:.2%}")

        prediction_df = pd.DataFrame({
            'Labels': labels,
            'Scores': scores
        })
        st.line_chart(prediction_df.set_index('Labels'))


st.markdown("---")
st.subheader("📍 Reforestation Mapping")

with st.expander("Add Location Coordinates"):
    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=22.8421673)
    lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=86.1018496)

if st.button("🌍 Show Location on Map"):
    m = folium.Map(location=[lat, lon], zoom_start=15)
    folium.Marker([lat, lon], popup='Predicted Plant Location').add_to(m)
    folium_static(m)
