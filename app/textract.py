import gradio as gr
from pprint import pprint
from collections import defaultdict
import boto3
import io
import pdfplumber
from io import BytesIO
from PIL import Image
import pandas as pd
import numpy as np
import google.generativeai as genai
import re
import os
import tempfile

from PIL import Image



# Your AWS session and client setup
session = boto3.Session()  



# Helper functions
def get_rows_columns_map(table_result, blocks_map):
    rows = {}
    for relationship in table_result['Relationships']:
        if relationship['Type'] == 'CHILD':
            for child_id in relationship['Ids']:
                cell = blocks_map[child_id]
                if cell['BlockType'] == 'CELL':
                    row_index = cell['RowIndex']
                    col_index = cell['ColumnIndex']
                    if row_index not in rows:
                        rows[row_index] = {}
                    rows[row_index][col_index] = get_text(cell, blocks_map)
    return rows


def get_text(result, blocks_map):
    text = ''
    if 'Relationships' in result:
        for relationship in result['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    word = blocks_map[child_id]
                    if word['BlockType'] == 'WORD':
                        text += word['Text'] + ' '
    return text.strip()


def generate_table_df(table_result, blocks_map):
    rows = get_rows_columns_map(table_result, blocks_map)
    table_data = []
    for _, cols in rows.items():
        row_data = [text for _, text in sorted(cols.items())]
        table_data.append(row_data)
    table_df = pd.DataFrame(table_data)
    table_df.columns = [f"Column {i+1}" for i in range(table_df.shape[1])]  # Add column headers
    return table_df



def get_table_results(image_bytes):
    client = session.client('textract', region_name='us-east-1')
    response = client.analyze_document(Document={'Bytes': image_bytes}, FeatureTypes=['TABLES'])
    blocks = response['Blocks']
    blocks_map = {block['Id']: block for block in blocks}
    table_blocks = [block for block in blocks if block['BlockType'] == "TABLE"]

    if not table_blocks:
        return "No Table Found"

    results = []
    for index, table in enumerate(table_blocks):
        table_df = generate_table_df(table, blocks_map)
        results.append({"table_df": table_df, "table_index": index + 1})
    return results



# Gradio Interface Function
def process_image_key_table(image):
    if isinstance(image, np.ndarray):  # Convert numpy array to PIL image
        image = Image.fromarray(image)
    image_bytes_io = BytesIO()
    image.save(image_bytes_io, format='PNG')
    image_bytes = image_bytes_io.getvalue()
    
    tables_data = get_table_results(image_bytes)
    if isinstance(tables_data, str):  # No tables found
        return "No Table Found"
    
    # Return only the first table's DataFrame for simplicity
    return tables_data[0]["table_df"]



def table_extraction(image):
    table_df = process_image_key_table(image)
    return table_df







#######================================####################################33
# Key-Value extraction functions (updated)

def get_kv_map(image_bytes):
    # Process the image using bytes (directly received as input)
    session = boto3.Session()
    client = session.client('textract', region_name='us-east-1')
    response = client.analyze_document(Document={'Bytes': image_bytes}, FeatureTypes=['FORMS'])

    # Get the text blocks
    blocks = response['Blocks']

    # Get key and value maps
    key_map = {}
    value_map = {}
    block_map = {}
    for block in blocks:
        block_id = block['Id']
        block_map[block_id] = block
        if block['BlockType'] == "KEY_VALUE_SET":
            if 'KEY' in block['EntityTypes']:
                key_map[block_id] = block
            else:
                value_map[block_id] = block

    return key_map, value_map, block_map


def get_kv_relationship(key_map, value_map, block_map):
    kvs = defaultdict(list)
    for block_id, key_block in key_map.items():
        value_block = find_value_block(key_block, value_map)
        key = get_text(key_block, block_map)
        val = get_text(value_block, block_map)
        kvs[key].append(val)
    return kvs


def find_value_block(key_block, value_map):
    for relationship in key_block['Relationships']:
        if relationship['Type'] == 'VALUE':
            for value_id in relationship['Ids']:
                value_block = value_map[value_id]
    return value_block


def get_text(result, blocks_map):
    text = ''
    if 'Relationships' in result:
        for relationship in result['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    word = blocks_map[child_id]
                    if word['BlockType'] == 'WORD':
                        text += word['Text'] + ' '
                    if word['BlockType'] == 'SELECTION_ELEMENT':
                        if word['SelectionStatus'] == 'SELECTED':
                            text += 'X '

    return text


def print_kvs(kvs):
    result = []
    for key, value in kvs.items():
        result.append(f"{key}: {value}")
    return result


def search_value(kvs, search_key):
    for key, value in kvs.items():
        if re.search(search_key, key, re.IGNORECASE):
            return value
    return "Key not found."





##################  trial addtion of llm##################33

google_api_key ="AIzaSyBS-YXsW415Q5LPbe_EOT731KeoUqrrbIU"

# Configuring the generativeai module with the obtained API key for authentication
genai.configure(api_key=google_api_key)

# Creating an instance of the GenerativeModel class with the model name 'gemini-pro-vision'
model = genai.GenerativeModel('gemini-1.5-flash')


def process_pdf_page(pdf_path, page_number, user_prompt):
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




def orchestrate_extraction(data_type, uploaded_file, uploaded_image, user_prompt="", page_number=0):
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
                generated_text = process_pdf_page(pdf_path, int(page_number), user_prompt)
                
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

with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("# Document Data Extraction")
        
        data_type_selector = gr.Radio(
            choices=["Table Extraction", "Key-Value Extraction", "LLM Enabled Text Extraction"],
            label="Select Extraction Type",
            interactive=True
        )
        
        with gr.Row():
            with gr.Column():
                file_input = gr.File(
                    label="Upload a PDF File (for LLM Extraction)", 
                    type="binary",
                    interactive=True
                )
            with gr.Column():
                image_input = gr.Image(
                    label="Upload an Image (for Table/KV Extraction)", 
                    type="pil",
                    interactive=True
                )
        
        with gr.Row():
            user_prompt = gr.Textbox(
                label="LLM Prompt", 
                placeholder="Enter your custom prompt here",
                interactive=True
            )
            page_number = gr.Number(
                label="Page Number (0-indexed)", 
                value=0, 
                minimum=0,
                interactive=True
            )
        
        # Prominent submit button
        submit_btn = gr.Button(
            "Submit for Extraction", 
            variant="primary", 
            size="lg"
        )
        
        with gr.Row():
            output_dataframe = gr.Dataframe(label="Extracted Table Data")
            output_textbox = gr.Textbox(
                label="Extraction Result",
                lines=5
            )
    
        # Clear button to reset the form
        clear_btn = gr.Button("Clear All", variant="secondary")
        
        # Connect the submit button to the extraction function
        submit_btn.click(
            fn=orchestrate_extraction,
            inputs=[
                data_type_selector,
                file_input,
                image_input,
                user_prompt,
                page_number
            ],
            outputs=[
                output_dataframe,
                output_textbox
            ]
        )
        
        # Clear functionality
        def clear_inputs():
            return {
                file_input: None,
                image_input: None,
                user_prompt: "",
                page_number: 0,
                output_dataframe: pd.DataFrame(),
                output_textbox: ""
            }
            
        clear_btn.click(
            fn=clear_inputs,
            inputs=[],
            outputs=[
                file_input,
                image_input,
                user_prompt,
                page_number,
                output_dataframe,
                output_textbox
            ]
        )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)
