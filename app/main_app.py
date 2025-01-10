
import os
import sys

cur_dir = os.getcwd()
parent_dir = os.path.realpath(os.path.join(os.path.dirname(cur_dir)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    sys.path.append(cur_dir)
sys.path.insert(1, ".")

from orchestrator.orchestrate import *
from PIL import Image
from io import BytesIO
import gradio as gr
import pandas as pd
import numpy as np

import io





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
