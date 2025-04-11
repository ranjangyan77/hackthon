import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import pandas as pd
import cv2
import folium
from streamlit_folium import folium_static

model = ResNet50(weights='imagenet')


st.title("🌱 Smart Plant Identifier & Reforestation Mapper")
st.write("📸 Upload an image or capture from your camera to identify plants and map their location.")


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
            st.warning("Capture cancelled")
            break
    cam.release()
    cv2.destroyAllWindows()


if st.button("📷 Capture Image"):
    capture_image()
    uploaded_file = 'captured_image.jpg'
else:
    uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

  
    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=3)[0]

    
    st.image(img, caption='Uploaded Image', use_column_width=True)

   
    labels = [label for (_, label, _) in decoded_preds]
    scores = [score for (_, _, score) in decoded_preds]

    
    st.write("🌿 **Predictions:**")
    for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        st.write(f"**{i + 1}:** {label} (Confidence: {score:.2f})")

   
    prediction_df = pd.DataFrame({
        'Labels': labels,
        'Scores': scores
    })
    st.line_chart(prediction_df.set_index('Labels')['Scores'])

    
    st.write("📍 **Map Location:**")
    lat = st.number_input("Enter Latitude", min_value=-90.0, max_value=90.0, value=22.8418033)
    lon = st.number_input("Enter Longitude", min_value=-180.0, max_value=180.0, value=86.102168)

   
    if st.button("🗺️ Show Location"):
        m = folium.Map(location=[lat, lon], zoom_start=15)
        folium.Marker(
            [lat, lon],
            popup=f"<b>Plant Identified:</b> {labels[0]}<br><b>Confidence:</b> {scores[0]:.2f}",
            tooltip="Click for details"
        ).add_to(m)
        
       
        folium.TileLayer('Stamen Toner').add_to(m)
        folium.TileLayer('Stamen Watercolor').add_to(m)
        
        folium_static(m)

