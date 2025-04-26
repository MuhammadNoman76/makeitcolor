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
import shutil

# Fix for Python 3.12 asyncio issue
if sys.version_info >= (3, 12) and sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def main():
    st.set_page_config(page_title="Image Colorization", layout="wide")
    st.title("Black & White to Color Image Converter")
    
    # Create examples directory if it doesn't exist
    examples_dir = "./examples"
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)
        
        example_images = ["image1jpg", "image.jpg", "image3.jpg"]
        for img in example_images:
            if os.path.exists(img):
                shutil.copy(img, os.path.join(examples_dir, img))
    
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
    
    # Example images section
    st.subheader("Try with Example Images")
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        if os.path.exists(os.path.join(examples_dir, "image1.jpg")):
            st.image(os.path.join(examples_dir, "image1.jpg"), caption="Example 1", use_container_width=True)
            if st.button("Colorize Example 1"):
                process_image(os.path.join(examples_dir, "image1.jpg"), img_colorization)
    
    with example_col2:
        if os.path.exists(os.path.join(examples_dir, "image2.jpg")):
            st.image(os.path.join(examples_dir, "image2.jpg"), caption="Example 2", use_container_width=True)
            if st.button("Colorize Example 2"):
                process_image(os.path.join(examples_dir, "image2.jpg"), img_colorization)
    
    with example_col3:
        if os.path.exists(os.path.join(examples_dir, "image3.jpg")):
            st.image(os.path.join(examples_dir, "image3.jpg"), caption="Example 3", use_container_width=True)
            if st.button("Colorize Example 3"):
                process_image(os.path.join(examples_dir, "image3.jpg"), img_colorization)
    
    st.markdown("---")
    st.subheader("Upload Your Own Image")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a black and white image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and display the original image
        original_image = Image.open(uploaded_file)
        
        # Create temporary file path for processing
        temp_path = "temp_input.jpg"
        original_image.save(temp_path)
        
        process_image(temp_path, img_colorization)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def process_image(image_path, img_colorization):
    # Display original image
    original_image = Image.open(image_path)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_container_width=True)
    
    # Colorize the image
    try:
        with st.spinner("Colorizing image..."):
            result = img_colorization(image_path)
            colorized_image = result['output_img']
            
            # Convert BGR to RGB for display
            colorized_image_rgb = cv2.cvtColor(colorized_image, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("Colorized Image")
                st.image(colorized_image_rgb, use_container_width=True)
            
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


if __name__ == "__main__":
    main()

st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
        <p>Developed by <a href="https://www.linkedin.com/in/muhammad-noman76/" target="_blank">Muhammad Noman</a> | 
        Contact: muhammadnomanshafiq76@gmail.com</p>
    </div>
""", unsafe_allow_html=True)