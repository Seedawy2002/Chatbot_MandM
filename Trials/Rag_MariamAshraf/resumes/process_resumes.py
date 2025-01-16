




# import os
# import pypdf
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import pickle

# # Function to extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     pdf_reader = pypdf.PdfReader(pdf_path)
#     text = ""
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text

# # Directory containing resumes
# resumes_dir = 'resumes'
# pdf_files = [f for f in os.listdir(resumes_dir) if f.endswith('.pdf')]
# chunks = []

# # Process each PDF file
# for pdf_file in pdf_files:
#     pdf_path = os.path.join(resumes_dir, pdf_file)
#     print(f"Processing {pdf_path}")
#     try:
#         # Extract text from PDF
#         text = extract_text_from_pdf(pdf_path)
        
#         # Create a Document object
#         doc = Document(page_content=text, metadata={'source': pdf_file})
        
#         # Split text into chunks
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         chunked_docs = text_splitter.split_documents([doc])
        
#         # Add chunks to the list
#         chunks.extend(chunked_docs)
#     except Exception as e:
#         print(f"Error processing {pdf_path}: {e}")
#         continue

# # Save chunks to a file
# with open('chunks.pkl', 'wb') as f:
#     pickle.dump(chunks, f)

# print("Resumes processed and chunks saved to chunks.pkl")




from pdfminer.high_level import extract_text
from langchain.schema import Document  # Corrected import
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pickle

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

resumes_dir = 'resumes'
pdf_files = [f for f in os.listdir(resumes_dir) if f.endswith('.pdf')]
chunks = []

for pdf_file in pdf_files:
    pdf_path = os.path.join(resumes_dir, pdf_file)
    print(f"Processing {pdf_path}")
    try:
        text = extract_text_from_pdf(pdf_path)
        doc = Document(page_content=text, metadata={'source': pdf_file})
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunked_docs = text_splitter.split_documents([doc])
        chunks.extend(chunked_docs)
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        continue

with open('chunks.pkl', 'wb') as f:
    pickle.dump(chunks, f)

