import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, VideoTransformerBase
import av
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import register_keras_serializable
from keras import backend as K
import matplotlib.pyplot as plt
import cv2
from io import BytesIO
import os
import time
import json

background_ratio = 0.8
wound_ratio = 0.2
class_weights = {0: 1.0, 1: background_ratio / wound_ratio}


def dice_coef(y_true, y_pred, smooth=10e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def weighted_dice_coef(y_true, y_pred, smooth=10e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    weights = K.gather(K.constant(list(class_weights.values())), K.cast(y_true_f, dtype='int32'))
    intersection = K.sum(y_true_f * y_pred_f * weights)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f * weights) + K.sum(y_pred_f * weights) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred, smooth=10e-6):
    return 1 - dice_coef(y_true, y_pred, smooth)


def weighted_dice_coef_loss(y_true, y_pred, smooth=10e-6):
    dice_coef = weighted_dice_coef(y_true, y_pred, smooth)
    loss = 1 - dice_coef
    return loss


def iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + 1.0) / (union + 1.0)
    return iou


def iou_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou


def combined_dice_iou_loss(y_true, y_pred, smooth=1e-6, dice_weight=0.5, iou_weight=0.5, wound_weight=0.75):
    unweighted_dice_loss = dice_coef_loss(y_true, y_pred, smooth)
    weighted_dice_loss = weighted_dice_coef_loss(y_true, y_pred, smooth)
    combined_dice_loss = wound_weight * weighted_dice_loss + (1 - wound_weight) * unweighted_dice_loss

    iou_loss_val = iou_loss(y_true, y_pred, smooth)

    combined_loss = dice_weight * combined_dice_loss + iou_weight * iou_loss_val
    return combined_loss
# Load the trained model
@st.cache_resource
def load_model():
    model_path = 'june11v9.keras'  # Ensure this path is correct for your Streamlit deployment
    model = tf.keras.models.load_model(model_path, custom_objects={
        'dice_coef': dice_coef,
        'weighted_dice_coef': weighted_dice_coef,
        'dice_coef_loss': dice_coef_loss,
        'weighted_dice_coef_loss': weighted_dice_coef_loss,
        'iou': iou,
        'iou_loss': iou_loss,
        'combined_dice_iou_loss': combined_dice_iou_loss
    })
    return model

def segment_wound(image, model, threshold):
    start_time = time.time()
    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make predictions
    prediction = model.predict(image)

    # Convert the prediction to a binary mask
    binary_mask = (prediction[0, :, :, 0] > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Find the contours of the predicted wound area
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image for visualization
    image_with_contour = image[0].copy()

    # Draw the contours on the image
    image_with_contour = cv2.drawContours(image_with_contour, contours, -1, (0, 255, 0), 2)

    end_time = time.time()
    processing_time = end_time - start_time
    return image[0], image_with_contour, opened, prediction, processing_time

# Streamlit app
st.title('Wound Segmentation App')
st.write("Delineates acute traumatic injuries from either uploaded images, webcam captures, or through a live video feed via webcam.")

# Input Options
input_source = st.radio("Select input source:", ("Upload Image", "Webcam Capture"))
st.subheader('Prediction Confidence')
threshold = st.slider('Confidence', 0.001, 0.999, 0.999, 0.001, format="%.3f")

if st.button('Reset to 99.9%'):
    threshold = 0.999

# Load model once (cached)
model = load_model()

if input_source == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        original, contour, mask, prediction, processing_time = segment_wound(image, model, threshold)
        original = (original * 255).astype(np.uint8)
        contour = (contour * 255).astype(np.uint8)
        mask = (mask * 255).astype(np.uint8)

        # Display results in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(original, caption='Original Image', use_column_width=True)
        with col2:
            st.image(contour, caption='Predicted Contour', use_column_width=True)
        with col3:
            st.image(mask, caption='Predicted Mask', use_column_width=True)

        st.write(f"Processing Time: {processing_time:.3f} seconds")
        # Calculate and display metrics
        st.subheader('Segmentation Metrics')
        total_pixels = original.shape[0] * original.shape[1]
        wound_pixels = np.count_nonzero(mask == 255)
        wound_percentage = (wound_pixels / total_pixels) * 100

        st.write(f"Wound Area: {wound_pixels} pixels")
        st.write(f"Percentage of Wound Area: {wound_percentage:.2f}%")

elif input_source == "Webcam Capture":
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        if image is not None:
            original, contour, mask, prediction, processing_time = segment_wound(image, model, threshold)
            original = (original * 255).astype(np.uint8)
            contour = (contour * 255).astype(np.uint8)
            mask = (mask * 255).astype(np.uint8)

            # Display results in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(original, caption='Original Image', use_column_width=True)
            with col2:
                st.image(contour, caption='Predicted Contour', use_column_width=True)
            with col3:
                st.image(mask, caption='Predicted Mask', use_column_width=True)

            st.write(f"Processing Time: {processing_time:.3f} seconds")
            # Calculate and display metrics
            st.subheader('Segmentation Metrics')
            total_pixels = original.shape[0] * original.shape[1]
            wound_pixels = np.count_nonzero(mask == 255)
            wound_percentage = (wound_pixels / total_pixels) * 100

            st.write(f"Wound Area: {wound_pixels} pixels")
            st.write(f"Percentage of Wound Area: {wound_percentage:.2f}%")


st.sidebar.title('About')
st.sidebar.info('This app demonstrates wound segmentation using a deep learning model specifically designed for practical applications of Computer Vision in emergency medical devices. Upload an image, take a picture, or use the live webcam feed to see the segmentation results.')
st.sidebar.info('Prediction confidence is automatically set to 99.9% unless adjusted with the slider.')