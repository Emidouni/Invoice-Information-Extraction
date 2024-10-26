import pytesseract
import os
from PIL import Image
import easyocr
import spacy
import re
import cv2


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
nlp = spacy.load("en_core_web_trf")


def convert_to_300_dpi(input_image_path, output_folder, output_image_name):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder '{output_folder}' created.")

    # Open the image
    image = Image.open(input_image_path)
    
    # Full path to save the output image in the new folder
    output_image_path = os.path.join(output_folder, output_image_name)
    
    # Save the image with 300 DPI
    image.save(output_image_path, dpi=(500, 500))

    # Display a message indicating why 300 DPI is important for OCR
    print(f"Image saved as '{output_image_path}' with 300 DPI. "
          "This resolution is important for better OCR accuracy.")
    return output_image_path

def text_extraction(image):
    # Initialize the OCR reader
    text_reader = easyocr.Reader(['en'])
    
    # Perform OCR on the input image
    results = text_reader.readtext(image)
    extracted_text = ""
    
    # Iterate through the results and concatenate each text segment with a newline
    for (bbox, text_segment, prob) in results:
        extracted_text += text_segment + "\n"  # Add each text segment and a newline
    
    # Return all the recognized text with line breaks
    return extracted_text


def extract_earliest_date(text):
    """
    Extracts the earliest date from the given text.
    
    The function uses spaCy to find date entities in the text. 
    If no date entities are found, it falls back to regex to find 
    dates in various formats. If multiple dates are found, 
    it returns the earliest one.

    Args:
    text (str): The input text from which to extract dates.

    Returns:
    list: A list of dates found or a message if no dates are found.
    """
    # Split the text into lines
    lines = text.split('\n')

    # Collect all extracted dates
    extracted_dates = []
    
    # Regex patterns to match various date formats
    date_patterns = [
        r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b',  # MM/DD/YY or MM/DD/YYYY
        r'\b(\d{1,2}-\d{1,2}-\d{2,4})\b'   # MM-DD-YY or MM-DD-YYYY
    ]

    for line in lines:
        # Use spaCy to find date entities
        doc = nlp(line)
        date_entities = [ent.text for ent in doc.ents if ent.label_ == "DATE"]

        # If spaCy finds dates, add them to the list
        if date_entities:
            extracted_dates.extend(date_entities)
        else:
            # Use regex to find dates in the patterns specified
            for date_pattern in date_patterns:
                matches = re.findall(date_pattern, line)
                extracted_dates.extend(matches)  # Add regex matches to the list

    return extracted_dates

def process_image(image_path, output_folder):
    output_image_path = convert_to_300_dpi(image_path, output_folder, 'temp_image_300dpi.png')
    image = cv2.imread(output_image_path)
    # cropped_image = detect_largest_angled_text_roi(image)
    # corrected_image = rotation_correction(cropped_image, image)
    if image is not None:
        text = text_extraction(image)
        extracted_dates = extract_earliest_date(text)
        os.remove(output_image_path)  # Supprimer l'image temporaire
        return extracted_dates
    else:
        return []


