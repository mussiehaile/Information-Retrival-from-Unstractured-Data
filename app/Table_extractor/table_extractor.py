

import os
import sys

cur_dir = os.getcwd()
parent_dir = os.path.realpath(os.path.join(os.path.dirname(cur_dir)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    sys.path.append(cur_dir)
sys.path.insert(1, ".")


from PIL import Image
import pandas as pd
import numpy as np
from io import BytesIO
import boto3





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