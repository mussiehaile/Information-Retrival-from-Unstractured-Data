
# **Image and PDF Text Extraction Tool**

## **Overview**
This project provides a comprehensive tool to extract text and organize information from images or PDFs. Users can select from three distinct extraction methods tailored to different use cases:

1. **Table Extraction**: Extract tabular data, ideal for documents with structured tables.
2. **Key-Value Extraction**: Retrieve key-value pairs, suitable for invoices, forms, or similar data.
3. **LLM-Powered Extraction**: Leverage a large language model (LLM) to extract and organize information using custom prompts, designed for a flexible and contextual understanding of data.

## **Video samples of the App**
- **Table extraction**:
[table.webm](https://github.com/user-attachments/assets/41c00b07-8ff7-44c3-b59d-5e224d0861ab)
- **Key-value extraction**:
  [key_value.webm](https://github.com/user-attachments/assets/e45fb248-8ad4-410f-897e-d9a187234bbb)

- **LLM-enabled extraction**:
  
[llm.webm](https://github.com/user-attachments/assets/307419c6-db98-40fe-beb9-c8a0ef254eb1)

## **Features**
- **Image to PDF Conversion**: Automatically converts uploaded images into PDFs for seamless processing.
- **Text Extraction**: Extract text from images or PDFs using powerful tools like AWS Textract and PDFParser.
- **User Choice**: Interactive interface to choose between table extraction, key-value extraction, or LLM-based processing.
- **Organized Output**: Presents extracted information in an organized and user-friendly format.

## **Tech Stack**
- **Python**: Core programming language.
- **AWS Textract**: For extracting structured text, tables, and key-value pairs from documents.
- **PDFParser**: For parsing and extracting text from PDF files.
- **Google Multimodal Language Model**: For contextual and flexible extraction using advanced prompts.
- **Gradio**: Interactive interface for displaying and interacting with extracted data.

## **Usage**
### **Prerequisites**
Ensure the following are installed and configured on your system:
- Python 3.8+
- AWS CLI (configured with valid credentials for AWS Textract)
- Required Python libraries (install via `requirements.txt`)

### **Installation**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/mussiehaile/Information-Retrival-from-Unstractured-Data
   cd app
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### **Running the Application**
1. **Start the application**:
   ```bash
   python main_app.py
   ```
2. **Upload an image or PDF file**.
3. **Select one of the extraction methods (Table, Key-Value, or LLM-Powered)**.
4. **View the extracted and organized output via the Radio interface**.

### **Configuration**
- Update your AWS credentials in `~/.aws/credentials` or configure using the AWS CLI.
- Modify prompts or extraction parameters in `config.py` as needed.

## **Example Use Cases**
- **Table Extraction**: Parsing tables from financial statements or reports.
- **Key-Value Extraction**: Extracting data from invoices, forms, or contracts.
- **LLM-Powered Extraction**: Analyzing unstructured text, generating summaries, or contextual extraction from complex documents.

## **Contribution**
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with a clear description of the changes.

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

## **Contact**
For questions or support, contact **[Your Name/Your Organization]** at **[musshaile@gmail.com]**.

