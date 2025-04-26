import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from huggingface_hub import snapshot_download
import io
import sys
import asyncio

# Fix for Python 3.12 asyncio issue
if sys.version_info >= (3, 12) and sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def main():
    st.set_page_config(page_title="Image Colorization", layout="wide")
    st.title("Black & White to Color Image Converter")
    
    # Check if model is already downloaded
    model_dir = "./makeitcolor"
    if not os.path.exists(model_dir):
        with st.spinner("Downloading model (this might take a few minutes)..."):
            snapshot_download(repo_id="muhammadnoman76/makeitcolor", local_dir=model_dir, repo_type="model")
    
    # Initialize the colorization pipeline
    try:
        img_colorization = pipeline(Tasks.image_colorization, model=model_dir)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a black and white image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and display the original image
        original_image = Image.open(uploaded_file)
        
        # Create temporary file path for processing
        temp_path = "temp_input.jpg"
        original_image.save(temp_path)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(original_image, use_container_width=True)  # Updated parameter
        
        # Colorize the image
        try:
            with st.spinner("Colorizing image..."):
                result = img_colorization(temp_path)
                colorized_image = result['output_img']
                
                # Convert BGR to RGB for display
                colorized_image_rgb = cv2.cvtColor(colorized_image, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.subheader("Colorized Image")
                    st.image(colorized_image_rgb, use_container_width=True)  # Updated parameter
                
                # Add download button for colorized image
                colorized_pil = Image.fromarray(colorized_image_rgb)
                buf = io.BytesIO()
                colorized_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Colorized Image",
                    data=byte_im,
                    file_name="colorized_image.png",
                    mime="image/png"
                )
        except Exception as e:
            st.error(f"Error during colorization: {e}")
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    main()