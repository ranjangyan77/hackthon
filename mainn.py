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

# Initialize the ResNet50 model
model = ResNet50(weights='imagenet')

# Title of the app
st.title("Smart Plant Identifier & Reforestation Mapper")
st.write("Upload an image or capture from camera to get predictions.")

# Function to capture an image from the camera
def capture_image():
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture image")
            break
        cv2.imshow("Press 'c' to capture, 'q' to quit", frame)
        key = cv2.waitKey(1)
        if key == ord('c'):
            cv2.imwrite('captured_image.jpg', frame)
            break
        elif key == ord('q'):
            print("Capture cancelled")
            break
    cam.release()
    cv2.destroyAllWindows()

# Button to capture an image
if st.button("Capture Image"):
    capture_image()
    uploaded_file = 'captured_image.jpg'
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Process the uploaded image
if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make predictions
    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=3)[0]

    # Display the uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Extract labels and scores from predictions
    labels = [label for (_, label, _) in decoded_preds]
    scores = [score for (_, _, score) in decoded_preds]

    # Display predictions
    st.write("Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        st.write(f"{i + 1}: {label} ({score:.2f})")

    # Create a DataFrame for predictions
    prediction_df = pd.DataFrame({
        'Labels': labels,
        'Scores': scores
    })

    # Display a line chart of prediction scores
    st.line_chart(prediction_df.set_index('Labels')['Scores'])

    # Input for location
    st.write("Location:")
    lat = st.number_input("Enter Latitude", min_value=-90.0, max_value=90.0, value=22.8421673)
    lon = st.number_input("Enter Longitude", min_value=-180.0, max_value=180.0, value=86.1018496)

    # Create an interactive Folium map
    if st.button("Show Location"):
        m = folium.Map(location=[lat, lon], zoom_start=15)
        folium.Marker([lat, lon], popup='Predicted Location').add_to(m)
        folium_static(m)
