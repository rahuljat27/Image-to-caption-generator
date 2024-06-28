import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
import numpy as np
import io
from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast, ViTImageProcessor

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_data
def get_model():
    model = VisionEncoderDecoderModel.from_pretrained("Rahuljat27/Image-Caption-Generator")
    tokenizer = GPT2TokenizerFast.from_pretrained("Rahuljat27/Image-Caption-Generator")
    return model, tokenizer

model, tokenizer = get_model()
image_processor = ViTImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")

def generate_caption(image_array):
    try:
        image = Image.fromarray(image_array).convert("RGB")
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)

        # Generate the caption
        output_ids = model.generate(pixel_values, max_length=32, num_beams=4)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return caption
    except UnidentifiedImageError:
        return "Error: Cannot identify image file. Please check the uploaded image."
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit app
st.title("Image Captioning with VisionEncoderDecoderModel")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read the uploaded file as a numpy array
        image = Image.open(uploaded_file)
        image_array = np.array(image)

        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Generating caption...")

        caption = generate_caption(image_array)
        st.write("Generated Caption:", caption)
    except Exception as e:
        st.write(f"Error: {str(e)}")
