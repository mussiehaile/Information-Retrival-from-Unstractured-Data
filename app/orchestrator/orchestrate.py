
import os
import sys

cur_dir = os.getcwd()
parent_dir = os.path.realpath(os.path.join(os.path.dirname(cur_dir)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    sys.path.append(cur_dir)
sys.path.insert(1, ".")

from app.Key_value_Extractor.key_value import *
from app.M_LLM_Extractor.mllm_extractor import*
from app.Table_extractor.table_extractor import *
from PIL import Image
from io import BytesIO
import pandas as pd
import numpy as np
import tempfile
import io



## create model instance
model = create_connection(
    ConfigStaticFiles.google_gemini_api_key,
    ConfigStaticFiles.mllm_name

)

def orchestrate_extraction(data_type, 
                           uploaded_file,
                             uploaded_image, 
                             user_prompt="",
                               page_number=0
                               ):
    # Input validation
    if data_type is None:
        return pd.DataFrame(), "Please select an extraction type."
    
    try:
        if data_type == "Table Extraction" or data_type == "Key-Value Extraction":
            if uploaded_image is None:
                return pd.DataFrame(), "Please upload an image for extraction."
            
            # Convert uploaded image to PIL Image
            if isinstance(uploaded_image, (np.ndarray, bytes, str)):
                image = Image.open(io.BytesIO(uploaded_image)) if isinstance(uploaded_image, (bytes, str)) else Image.fromarray(uploaded_image)
            elif isinstance(uploaded_image, Image.Image):
                image = uploaded_image  # Already in PIL format
            else:
                raise ValueError(f"Unsupported image type: {type(uploaded_image)}")

            if data_type == "Table Extraction":
                try:
                    table_df = table_extraction(image)  # Your table extraction function
                    return table_df, "Table extraction completed"
                except Exception as e:
                    return pd.DataFrame(), f"Table extraction error: {str(e)}"

            else:  # Key-Value Extraction
                try:
                    # Convert PIL image to bytes
                    image_bytes_io = io.BytesIO()
                    image.save(image_bytes_io, format='PNG')
                    image_bytes = image_bytes_io.getvalue()

                    key_map, value_map, block_map = get_kv_map(image_bytes)
                    kvs = get_kv_relationship(key_map, value_map, block_map)
                    kvs_result = "\n".join([f"{key}: {', '.join(value)}" for key, value in kvs.items()])
                    return pd.DataFrame(), kvs_result
                except Exception as e:
                    return pd.DataFrame(), f"Key-Value extraction error: {str(e)}"

        elif data_type == "LLM Enabled Text Extraction":
            if uploaded_file is None:
                return pd.DataFrame(), "Please upload a PDF file for extraction."
            
            try:
                # Handle the PDF file
                if isinstance(uploaded_file, dict) and 'path' in uploaded_file:
                    pdf_path = uploaded_file['path']
                else:
                    # Create a temporary file to save the PDF content
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                        if isinstance(uploaded_file, bytes):
                            temp_pdf.write(uploaded_file)
                        elif hasattr(uploaded_file, 'read'):
                            temp_pdf.write(uploaded_file.read())
                        pdf_path = temp_pdf.name

                # Verify it's a PDF
                if not pdf_path.lower().endswith('.pdf'):
                    return pd.DataFrame(), "Invalid file type. Please upload a PDF."

                # Process the PDF
                generated_text = process_pdf_page(pdf_path, int(page_number), user_prompt,model)
                
                # Clean up temporary file if created
                if 'temp_pdf' in locals():
                    os.unlink(pdf_path)
                
                return pd.DataFrame(), generated_text

            except Exception as e:
                if 'temp_pdf' in locals():
                    os.unlink(pdf_path)
                return pd.DataFrame(), f"PDF processing error: {str(e)}"

    except Exception as e:
        return pd.DataFrame(), f"Error during processing: {str(e)}"

    return pd.DataFrame(), "Invalid option selected."