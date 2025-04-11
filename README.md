🌱 Smart Plant Identifier & Reforestation Mapper
A web-based app built with Streamlit that uses deep learning (ResNet50) to identify plants from images and map their locations for reforestation tracking.

<!-- Replace with your own screenshot or demo GIF -->

🚀 Features
📸 Upload or capture plant images using your webcam.

🤖 Identify plant species using the ResNet50 model pretrained on ImageNet.

📊 Visualize top-3 predictions with confidence scores.

🌍 Map plant locations using interactive Folium maps.

🧭 Supports manual input of GPS coordinates for mapping.

🛠️ Tech Stack
Frontend: Streamlit

Model: ResNet50 from tensorflow.keras.applications

Image Processing: OpenCV, PIL, NumPy

Mapping: Folium + streamlit-folium

📦 Installation
Clone the repository

bash
Copy
Edit
git clone https://github.com/your-username/plant-identifier-mapper.git
cd plant-identifier-mapper
Create and activate a virtual environment (optional but recommended)

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
streamlit run app.py
🧪 Example Use Case
Launch the app.

Upload an image of a plant or capture using webcam.

Let the model identify the plant species.

Enter latitude & longitude manually.

Visualize the location on an interactive map.

📁 Project Structure
bash
Copy
Edit
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── captured_image.jpg      # Captured image placeholder
✅ To-Do
 Integrate GPS auto-detection (e.g., via HTML5 geolocation).

 Save mapped data to a cloud database (e.g., Firebase, PostgreSQL).

 Improve model accuracy with a fine-tuned plant-specific CNN.

 Add historical view of reforestation markers.

🤝 Contributing
Contributions are welcome! Open issues, fork the repo, submit pull requests – all appreciated.

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

🌐 Links
📚 Streamlit Documentation

📘 ResNet50 Keras Docs

📍 Folium Docs

