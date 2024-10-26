import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from utils import convert_to_300_dpi, extract_earliest_date, text_extraction, easyocr, spacy

# Local path to the model (synchronized from OneDrive)
model_path = r'C:/Users/eyami/OneDrive/MODEL_stream/' 
model_checkpoint = "microsoft/swin-tiny-patch4-window7-224"

# Load NLP model and OCR reader
nlp = spacy.load("en_core_web_trf")
ocr_reader = easyocr.Reader(['en'])

# Paths and model checkpoints
processed_folder = "processed_images"
output_image_name = "output_image_300dpi.png"

# Load the image processor and model
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
model = AutoModelForImageClassification.from_pretrained(model_path)

# Normalization based on the processor's provided values
normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

# Determine the image size
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
else:
    size = (image_processor.size["shortest_edge"], image_processor.size["shortest_edge"])

# Image preprocessing function
def preprocess(image):
    return Compose([
        Resize(size),
        ToTensor(),
        normalize,
    ])(image)

# Streamlit Interface
st.title("Data Extraction and Image Classification")

# Instructions for the user
st.markdown("## Instructions")
st.markdown("""
- Drag and drop images or select them directly from your computer.
- Supported image types: PNG, JPG, JPEG.
- You can select an image to process by clicking on the thumbnail.
""")

# Upload images
uploaded_files = st.file_uploader("Choose images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

# Application state management for the selected image
if 'selected_image_index' not in st.session_state:
    st.session_state.selected_image_index = 0
if 'predicted_class' not in st.session_state:
    st.session_state.predicted_class = None
if 'selected_info' not in st.session_state:
    st.session_state.selected_info = None
if 'extracted_info' not in st.session_state:
    st.session_state.extracted_info = None

# If images are uploaded
if uploaded_files:
    st.markdown("## Select an Image by Clicking on its Thumbnail")
    
    # Display image thumbnails
    num_columns = 4
    columns = st.columns(num_columns)
    
    for index, uploaded_file in enumerate(uploaded_files):
        img = Image.open(uploaded_file)
        img.thumbnail((150, 150))
        
        col = columns[index % num_columns]
        if col.button(f"Select {uploaded_file.name}"):
            st.session_state.selected_image_index = index
            st.session_state.extracted_info = None  # Reset extracted information for a new selection
            
        col.image(img, use_column_width=True)

    # Display the selected image
    selected_image_file = uploaded_files[st.session_state.selected_image_index]
    selected_image = Image.open(selected_image_file)
    st.image(selected_image, caption=f"Selected Image: {selected_image_file.name}", use_column_width=True)
    
    # Process and predict image class
    if st.button("Process Selected Image"):
        st.write(f"Processing image: {selected_image_file.name}")
        
        # Preprocess the image
        pixel_values = preprocess(selected_image.convert("RGB")).unsqueeze(0)

        # Perform prediction
        with torch.no_grad():
            try:
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()

                # Store the predicted class in session state
                st.session_state.predicted_class = model.config.id2label[predicted_class_idx]
                st.write(f"Predicted class: {st.session_state.predicted_class}")

            except Exception as e:
                st.error(f"Error during prediction: {e}")

    # Perform information extraction if the predicted class is "invoice"
    if st.session_state.predicted_class and st.session_state.predicted_class.lower() == "invoice":
        processed_image_path = convert_to_300_dpi(selected_image_file, processed_folder, output_image_name)
        text = text_extraction(processed_image_path)

        # Dropdown for information selection
        st.session_state.selected_info = st.selectbox(
            "Select information to extract:",
            ["", "Date", "Client Name", "Client Address"]
        )

        # Display relevant extracted information based on selection
        if st.session_state.selected_info == "Date":
            # Extract dates
            extracted_dates = extract_earliest_date(text)
            st.session_state.extracted_info = extracted_dates
        elif st.session_state.selected_info == "Client Name":
            # Placeholder for client name extraction
            st.session_state.extracted_info = "Extracted Client Name"
        elif st.session_state.selected_info == "Client Address":
            # Placeholder for client address extraction
            st.session_state.extracted_info = "Extracted Client Address"

        # Display the extracted information if available
        if st.session_state.extracted_info:
            st.markdown("### Extracted Required Information")
            st.text_area("Extracted Data:", value=st.session_state.extracted_info, height=100)

else:
    st.write("No images uploaded yet.")

