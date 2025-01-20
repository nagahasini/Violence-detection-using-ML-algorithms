import streamlit as st
import cv2
import numpy as np
import tempfile
from tensorflow.keras.models import load_model
from collections import deque

# Load the pre-trained model
model = load_model('MobBiLSTM_model.h5')

# Class names for prediction
CLASSES_LIST = ["NonViolence", "Violence"]

# Set image dimensions, sequence length, and frame skip rate
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
FRAME_SKIP_RATE = 5  # Adjust this to control how many frames are skipped

def predict_frames_streamlit(video_file_path, SEQUENCE_LENGTH, frame_skip_rate=FRAME_SKIP_RATE):
    video_reader = cv2.VideoCapture(video_file_path)
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''
    frame_count = 0  # Counter to skip frames

    # Process video frames
    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break

        frame_count += 1
        # Skip frames based on the frame_skip_rate
        if frame_count % frame_skip_rate != 0:
            continue

        # Resize and normalize the frame
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            # Predict using the model
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]

        # Display the prediction on each frame
        display_frame = frame.copy()
        color = (0, 0, 255) if predicted_class_name == "Violence" else (0, 255, 0)
        label_text = f"{predicted_class_name}"

        # Add text to the frame
        cv2.putText(display_frame, label_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        # Show the frame with prediction in Streamlit
        st.image(display_frame, channels="BGR")
        
    video_reader.release()

# Streamlit File Upload and Prediction Interface
st.title("Violence Detection in Video - Frame by Frame")

uploaded_video = st.file_uploader("Upload a video for frame-by-frame prediction", type=["mp4"])

if uploaded_video is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        video_path = temp_file.name
        temp_file.write(uploaded_video.read())
    
    # Perform frame-by-frame prediction and display frames with predictions
    st.write("Processing and analyzing each frame...")
    predict_frames_streamlit(video_path, SEQUENCE_LENGTH, FRAME_SKIP_RATE)