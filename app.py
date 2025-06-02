import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Load your trained YOLOv5 checkpoint
#    Make sure you have ultralytics installed: pip install ultralytics
model = YOLO('/mnt/data/best.pt')

st.title("Car Model Detection")
st.write("Upload an image of a car below. If you’d like, specify a target model (class) to check for in the image.")

# 2. File uploader for the user’s image
uploaded_file = st.file_uploader("Upload a car image", type=['jpg', 'jpeg', 'png'])

# 3. Optional text input for a particular car‐model name
target_model = st.text_input(
    "If you want to detect a specific car model (class), enter its name here:",
    placeholder="e.g., sedan, hatchback, SUV, etc."
)

if uploaded_file:
    # 4a. Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("---")

    # 4b. Convert to NumPy array (YOLOv5 expects an array or file path)
    img_array = np.array(image)

    # 5. Run inference
    results = model.predict(source=img_array, conf=0.25)  # You can adjust confidence threshold

    # We’ll only use the first result (common when you pass a single image)
    r = results[0]
    boxes = r.boxes  # a list of Box objects

    if len(boxes) == 0:
        st.write("No objects detected in the image.")
    else:
        # 6a. Draw bounding boxes onto the image
        annotated_numpy = r.plot()  # returns a NumPy array with drawn boxes
        st.image(annotated_numpy, caption="Detected Objects", use_column_width=True)

        # 6b. Collect the class names from detections
        detected_classes = [model.names[int(x.cls)] for x in boxes]

        if target_model.strip():
            # 7a. If the user specified a target_model, filter by that class
            filtered = [
                cls for cls in detected_classes
                if cls.lower() == target_model.strip().lower()
            ]
            if filtered:
                st.success(f"✅ Detected the specified model: **{target_model.strip()}** ({len(filtered)} instance(s)).")
            else:
                st.error(f"❌ The specified model **{target_model.strip()}** was not detected.")
        else:
            # 7b. Otherwise, summarize all detected classes with their counts
            unique, counts = np.unique(detected_classes, return_counts=True)
            st.write("Detected classes (and counts):")
            for cls_name, count in zip(unique, counts):
                st.write(f"- {cls_name}: {count}")
