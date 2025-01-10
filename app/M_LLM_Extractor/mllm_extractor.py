


import os
import sys

cur_dir = os.getcwd()
parent_dir = os.path.realpath(os.path.join(os.path.dirname(cur_dir)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    sys.path.append(cur_dir)
sys.path.insert(1, ".")


from app.config.config import ConfigStaticFiles
import google.generativeai as genai
from PIL import Image
import numpy as np
import pdfplumber




## create connection with the model
def create_connection(api_key,model_name):

    # Configuring the generativeai module with the obtained API key for authentication
    genai.configure(api_key=api_key)

    # Creating an instance of the GenerativeModel class with the model name 'gemini-pro-vision'
    model = genai.GenerativeModel(model_name)

    return model




## processing engine ######
def process_pdf_page(pdf_path, page_number, user_prompt,model):
    """
    Extracts text and image from a PDF page, prepares a prompt, and generates organized content.

    Args:
        pdf_path (str): Path to the PDF file.
        page_number (int): Page number to process (0-indexed).
        user_prompt (str): The user's custom prompt to guide the LLM response.

    Returns:
        str: Generated content from the model.
    """
    try:
        # Open the PDF file
        with pdfplumber.open(pdf_path) as pdf:
            # Ensure the page number is valid
            if page_number < 0 or page_number >= len(pdf.pages):
                raise ValueError(f"Invalid page number: {page_number}. The PDF has {len(pdf.pages)} pages.")

            # Select the page
            page = pdf.pages[page_number]

            # Extract text
            income_text = page.extract_text()

            # Get the page image
            page_image = page.to_image(resolution=150)

            # Convert to array
            image_array = np.array(page_image.original)

            # Convert to a Pillow image
            pil_image = Image.fromarray(image_array)

        # Combine user prompt with extracted text
        prompt = f"""
        Here is the text extracted from the document:
        {income_text}

        {user_prompt}
        """

        # Generate content using a model (assuming a predefined `model` object)
        response = model.generate_content([prompt, pil_image])

        # Resolve the response to obtain the generated text
        response.resolve()

        # Return the generated text
        return response.text.strip()

    except Exception as e:
        return f"An error occurred: {str(e)}"


