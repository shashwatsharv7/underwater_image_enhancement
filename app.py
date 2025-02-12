import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from traditional_pipeline import full_pipeline, preprocess
from evaluation import calculate_metrics

TRAIN_B_PATH = "dataset/trainB"

def get_ground_truth_path(filename):
    return os.path.join(TRAIN_B_PATH, os.path.basename(filename))

def process_image(uploaded_file):
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        results = full_pipeline(temp_path)
        return results['final'], cv2.imread(temp_path)
    finally:
        os.remove(temp_path)

# Streamlit UI
st.set_page_config(page_title="Underwater Enhancer", layout="wide")
st.title("Underwater Image Enhancement System")

uploaded_file = st.file_uploader("Upload trainA image", type=["jpg", "png"])

if uploaded_file:
    final_img, raw_img = process_image(uploaded_file)
    gt_path = get_ground_truth_path(uploaded_file.name)
    
    # Three-column display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Noisy Input")
        st.image(raw_img, channels="BGR", use_column_width=True)
    with col2:
        st.subheader("Enhanced Result")
        st.image(final_img, use_column_width=True)
    with col3:
        if os.path.exists(gt_path):
            st.subheader("Ground Truth")
            st.image(preprocess(gt_path), use_column_width=True)
        else:
            st.warning("No GT available")
    
    # Metrics display
    if os.path.exists(gt_path):
        try:
            gt = preprocess(gt_path)
            metrics = calculate_metrics(final_img, gt)
            
            st.subheader("Performance Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['PSNR (dB)', 'SSIM'],
                'Value': [f"{float(metrics['PSNR']):.2f}", 
                         f"{float(metrics['SSIM']):.4f}"]
            }).set_index('Metric')
            
            st.table(metrics_df)
            
        except Exception as e:
            st.error(f"Metrics error: {str(e)}")