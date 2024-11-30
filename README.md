# Chatbot
![Uploading image.png…]()

## Overview  
This project focuses on creating a robust pipeline for handling textual and document-based data, leveraging state-of-the-art natural language processing (NLP) and OCR technologies. It consists of two primary workflows:
1. **Text Data Handling**: Processing, embedding, and vectorizing text for efficient information retrieval and chatbot functionalities.
2. **OCR Pipeline**: Extracting text from scanned documents and images for further processing.

## Features  
- **Text Embeddings**: Utilizes HuggingFace transformers for generating text embeddings.
- **Vector Stores**: Employs FAISS for efficient vector search and storage.
- **OCR**: Extracts text from PDFs and images using PyTesseract and PDF2Image.
- **Gradio Integration**: Interactive interface for processing and querying data.
- **LangChain Framework**: Chains NLP components for advanced workflows.

## File Descriptions  

### 1. `Chatbot_textData.ipynb`  
This notebook sets up a pipeline for processing raw text data:  
- **Dependencies**: Includes tools like LangChain, HuggingFace, FAISS, and Gradio.  
- **Functions**: 
  - Text splitting for better embedding granularity.
  - Embedding text using HuggingFace models.
  - Querying and searching through vectorized data.
- **Applications**: Useful for chatbot creation, document search, and semantic similarity tasks.

### 2. `Chatbot_OCR_RawData.ipynb`  
This notebook focuses on OCR tasks:  
- **Dependencies**: PyTesseract for OCR, PDF2Image for image conversion, FAISS, and LangChain.  
- **Functions**: 
  - Converts PDF pages to images and extracts text.
  - Embeds extracted text for vector-based search.
- **Applications**: Ideal for processing scanned documents or images and integrating OCR outputs into NLP workflows.

## Usage  

- Use **`Chatbot_textData.ipynb`** for:  
  - Loading and vectorizing raw text data.  
  - Building and testing a chatbot interface.  

- Use **`Chatbot_OCR_RawData.ipynb`** for:  
  - Extracting text from scanned documents or PDFs.  
  - Integrating OCR results with NLP pipelines.

## Technologies Used  
- **LangChain**  
- **HuggingFace Transformers**  
- **FAISS**  
- **PyTesseract**  
- **PDF2Image**  
- **Gradio**  

## Future Work  
- Trying more models
- Fintuning
- Adding prompts
- Enhance working on complicated CV files
- Allow the user to upload CVs

- Working on the docker file

## Contributors  
- Mariam Elseedawy
- Mariam Ashraf
