import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pdfplumber  
from docx import Document
from pathlib import Path

class CVProcessor:
    def __init__(self, watch_directory: str, processed_files_log: str = "processed_files.txt"):
        self.watch_directory = watch_directory
        self.processed_files_log = processed_files_log
        self.processed_files = self._load_processed_files()
        print(f"Initialized CV Processor for directory: {watch_directory}")
        print(f"Previously processed files: {len(self.processed_files)}")
    
    def _load_processed_files(self) -> set:
        if os.path.exists(self.processed_files_log):
            with open(self.processed_files_log, 'r') as f:
                return set(line.strip() for line in f)
        return set()
    
    def _save_processed_file(self, filename: str):
        with open(self.processed_files_log, 'a') as f:
            f.write(f"{filename}\n")
        self.processed_files.add(filename)
        print(f"Added {filename} to processed files log")
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        print(f"Attempting to extract text from PDF: {file_path}")
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                print(f"Successfully extracted {len(text)} characters from PDF")
                return text
        except Exception as e:
            print(f"Error processing PDF {file_path}: {str(e)}")
            print("Detailed error information:")
            import traceback
            traceback.print_exc()
            return ""

    def extract_text_from_docx(self, file_path: str) -> str:
        print(f"Attempting to extract text from DOCX: {file_path}")
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            print(f"Successfully extracted {len(text)} characters from DOCX")
            return text
        except Exception as e:
            print(f"Error processing DOCX {file_path}: {str(e)}")
            return ""

    def process_cv(self, file_path: str) -> dict:
        print(f"\nProcessing CV: {file_path}")
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            text_content = self.extract_text_from_pdf(file_path)
        elif file_ext in ['.docx', '.doc']:
            text_content = self.extract_text_from_docx(file_path)
        else:
            print(f"Unsupported file format: {file_ext}")
            return None
        
        if text_content:
            cv_data = {
                'file_path': file_path,
                'content': text_content,
                'processed_timestamp': time.time()
            }
            print(f"Successfully created CV data dictionary")
            # Print first 100 characters of extracted text for verification
            print(f"Preview of extracted text: {text_content[:100]}...")
            return cv_data
        return None

class CVHandler(FileSystemEventHandler):
    def __init__(self, processor: CVProcessor):
        self.processor = processor

    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = event.src_path
        print(f"\nFile system event detected: {event.event_type} - {file_path}")
        
        if file_path not in self.processor.processed_files:
            print(f"New CV detected: {file_path}")
            cv_data = self.processor.process_cv(file_path)
            
            if cv_data:
                print(f"Successfully processed: {file_path}")
                self.processor._save_processed_file(file_path)
                return cv_data
        else:
            print(f"File already processed: {file_path}")

def monitor_cv_directory(directory_path: str):
    abs_directory_path = os.path.abspath(directory_path)
    print(f"\nInitializing CV monitoring for directory: {abs_directory_path}")
    print(f"Directory exists: {os.path.exists(abs_directory_path)}")
    print(f"Directory contents: {os.listdir(abs_directory_path)}")
    
    processor = CVProcessor(abs_directory_path)
    event_handler = CVHandler(processor)
    observer = Observer()
    observer.schedule(event_handler, abs_directory_path, recursive=False)
    observer.start()
    
    try:
        print(f"\nStarted monitoring directory: {abs_directory_path}")
        print("Waiting for new files...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopped monitoring directory")
    observer.join()

if __name__ == "__main__":
    watch_dir = "/mnt/d/Users/mariam.ashraf/Desktop/Rag_MariamAshraf/resumes/resumes"
    monitor_cv_directory(watch_dir)