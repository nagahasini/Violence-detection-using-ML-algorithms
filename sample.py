import streamlit as st
import cv2
import numpy as np
import tempfile
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('MobBiLSTM_model.h5')


# Class names for prediction
CLASSES_LIST = ["NonViolence", "Violence"]

# Set image dimensions and sequence length for prediction
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16

def predict_video(video_file_path, SEQUENCE_LENGTH):
    video_reader = cv2.VideoCapture(video_file_path)

    # Get the width and height of the video
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Declare a list to store video frames
    frames_list = []

    # Get the number of frames in the video
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()

        if not success:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)

    # Make prediction using the model
    if len(frames_list) == SEQUENCE_LENGTH:
        predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis=0))[0]
        predicted_label = np.argmax(predicted_labels_probabilities)
        predicted_class_name = CLASSES_LIST[predicted_label]

        return predicted_class_name, predicted_labels_probabilities[predicted_label]
    else:
        return "Error", 0.0

# Streamlit File Upload and Prediction Interface
st.title('Violence Detection in Video')

# File uploader
uploaded_video = st.file_uploader("Upload a video for prediction", type=["mp4"])

if uploaded_video is not None:
    # Create a temporary directory and save the video file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        video_path = temp_file.name
        temp_file.write(uploaded_video.read())
    
    # Perform video prediction
    st.video(uploaded_video)  # Show the video in the Streamlit interface
    st.write("Analyzing the video...")

    # Call the function to predict violence or non-violence
    predicted_class, confidence = predict_video(video_path, SEQUENCE_LENGTH)

    # Display the result
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")