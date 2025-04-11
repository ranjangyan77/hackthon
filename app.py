import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import pandas as pd
import cv2
import folium
from streamlit_folium import folium_static
import streamlit_shadcn_ui as ui

# Initialize the ResNet50 model
@st.cache(allow_output_mutation=True)
def load_model():
    return ResNet50(weights='imagenet')

model = load_model()

# Title and description of the app
st.title("🌱 Smart Plant Identifier & Reforestation Mapper")
st.write("📸 Upload an image or capture from your camera to identify plants and map their location.")

# Function to capture an image from the camera
def capture_image():
    try:
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
        return 'captured_image.jpg'
    except Exception as e:
        st.error(f"Error capturing image: {e}")

# Button to capture an image
if st.button("📷 Capture Image"):
    uploaded_file = capture_image()
else:
    uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])

# Process the uploaded image
if uploaded_file is not None:
    try:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Make predictions using ResNet50
        preds = model.predict(x)
        decoded_preds = decode_predictions(preds, top=3)[0]

        # Display the uploaded image
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Extract labels and scores from predictions
        labels = [label for (_, label, _) in decoded_preds]
        scores = [score for (_, _, score) in decoded_preds]

        # Display predictions with better formatting
        st.write("🌿 **Predictions:**")
        for i, (imagenet_id, label, score) in enumerate(decoded_preds):
            st.write(f"**{i + 1}:** {label} (Confidence: {score:.2f})")

        # Create a DataFrame for predictions and display a chart
        prediction_df = pd.DataFrame({
            'Labels': labels,
            'Scores': scores
        })
        st.line_chart(prediction_df.set_index('Labels')['Scores'])

        # Input for location mapping
        st.write("📍 **Map Location:**")
        lat = st.number_input("Enter Latitude", min_value=-90.0, max_value=90.0, value=22.8418033)
        lon = st.number_input("Enter Longitude", min_value=-180.0, max_value=180.0, value=86.102168)

        # Create an interactive Folium map with enhanced features
        if st.button("🗺️ Show Location"):
            m = folium.Map(location=[lat, lon], zoom_start=15)
            folium.Marker(
                [lat, lon],
                popup=f"<b>Plant Identified:</b> {labels[0]}<br><b>Confidence:</b> {scores[0]:.2f}",
                tooltip="Click for details"
            ).add_to(m)
            
            # Add additional map layers (optional UX improvement)
            folium.TileLayer('Stamen Toner').add_to(m)
            folium.TileLayer('Stamen Watercolor').add_to(m)
            
            folium_static(m)

    except Exception as e:
        st.error(f"Error processing image: {e}")

# Streamlit Shadcn UI components
with ui.card(key="card1"):
    ui.element("span", children=["Email"], className="text-gray-400 text-sm font-medium m-1", key="label1")
    ui.element("input", key="email_input", placeholder="Your email")

    ui.element("span", children=["User Name"], className="text-gray-400 text-sm font-medium m-1", key="label2")
    ui.element("input", key="username_input", placeholder="Create a User Name")
    ui.element("button", text="Submit", key="button", className="m-1")

# Trigger button for alert dialog
trigger_btn = ui.button(text="Trigger Button", key="trigger_btn_1")
ui.alert_dialog(show=trigger_btn, title="Alert Dialog", description="This is an alert dialog", confirm_label="OK", cancel_label="Cancel", key="alert_dialog_1")
