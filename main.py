import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

model = ResNet50(weights='imagenet')

st.title("Image Classification with ResNet50")
st.write("Upload an image or capture from camera to get predictions.")


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


if st.button("Capture Image"):
    capture_image()
    uploaded_file = 'captured_image.jpg'
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

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

    
    st.write("Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        st.write(f"{i + 1}: {label} ({score:.2f})")

    
    prediction_df = pd.DataFrame({
        'Labels': labels,
        'Scores': scores
    })

    st.line_chart(prediction_df.set_index('Labels')['Scores'])

    
    st.write("Location:")
    lat = st.number_input("Enter Latitude", min_value=-90.0, max_value=90.0, value=37.7749)
    lon = st.number_input("Enter Longitude", min_value=-180.0, max_value=180.0, value=-122.4194)
    if st.button("Show Location"):
        st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}))
