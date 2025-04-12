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
import requests
from PIL import Image
import rasterio
from io import BytesIO


st.set_page_config(page_title="Plant Identifier & Reforestation Mapper", layout="wide")

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


def get_climate_data(lat, lon):
    """Fetch climate data from WorldClim API"""
    try:
        response = requests.get(f"https://worldclim.org/api/v2.1/climate?lat={lat}&lon={lon}")
        return response.json()['data']
    except:
        return None

def recommend_species(temp, precip, soil_type):
    """Tree recommendation logic based on research insights"""
    recommendations = []
    

    if temp > 10 and precip < 800:
        recommendations.extend([
            {'name': 'Norway Maple', 'traits': 'Drought-resistant, thermophilic'},
            {'name': 'Service Tree', 'traits': 'Climate-adapted, mixed-forest suitability'}
        ])
    
    
    if len(recommendations) > 1:
        recommendations.append({
            'name': 'Mixed Forest Package',
            'traits': '30% conifers + 70% broadleaf for stability'
        })
        
    return recommendations


def analyze_drone_image(img):
    """Process UAV imagery using OpenCV"""
    img_array = np.array(img.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)  

st.markdown("---")
st.subheader("🌳 Reforestation Recommendations")

if 'lat' in locals() and 'lon' in locals():
    climate_data = get_climate_data(lat, lon)
    
    if climate_data:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Average Temperature", f"{climate_data.get('temp', 'N/A')}°C")
            st.metric("Annual Precipitation", f"{climate_data.get('precip', 'N/A')}mm")
            
        with col2:
            recommended = recommend_species(
                climate_data.get('temp', 0),
                climate_data.get('precip', 0),
                climate_data.get('soil_type', 'N/A')
            )
            
            st.write("**Recommended Species:**")
            for species in recommended:
                st.markdown(f"- {species['name']}: _{species['traits']}_")
            
            
            st.warning("""
            **Ecological Best Practices:**
            - Minimum 15 species mix for biodiversity
            - Prioritize native species with <5% non-native
            """)
            

if uploaded_file and uploaded_file.type.startswith('image/'):
    img_pil = Image.open(uploaded_file)
    vegetation_count = analyze_drone_image(img_pil)
    st.write(f"Detected {vegetation_count} vegetation clusters in aerial imagery")
