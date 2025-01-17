# GenAI ChatBot for Resumes Screening

An AI-powered chatbot designed to streamline talent acquisition by helping identify and shortlist the best candidates for job vacancies. This chatbot leverages cutting-edge language models and intelligent data processing techniques to reduce time and effort while improving the quality of hiring decisions.

![image](https://github.com/user-attachments/assets/1ff5c573-968c-4679-9f80-7e8b74ef5e15)

![image](https://github.com/user-attachments/assets/5d9ec49b-d9df-431e-a1ad-a30ebe282467)


---

## **Features**
- **Resume Upload and Processing**:
  - Upload folders containing resumes in `.pdf` or `.docx` formats.
  - Automatically extracts, processes, and chunks text for efficient embedding and retrieval.
  - Generates embeddings and stores them in a vector database (`Pinecone`).

- **AI Chatbot for Candidate Querying**:
  - Users can interact with the chatbot to find the best candidates based on job-specific requirements.
  - Suggests and ranks candidates dynamically based on user-provided criteria.
  - Provides detailed candidate recommendations and comparisons.

- **Database Management**:
  - Ability to clear/reset the vector database for fresh processing.
  - Stores and retrieves context dynamically, supporting ongoing conversations.

- **User-Friendly Interface**:
  - Built with Gradio, offering an intuitive UI:
    - Tab for uploading folders and processing files.
    - Tab for interacting with the chatbot.
    - Dynamic chat interface with real-time updates.

---

## **Tech Stack**
- **Frontend**:
  - Gradio and Streamlit (2 versions): For building an interactive and responsive UI.
  
- **Backend**:
  - LangChain: Powers the logic for resume embedding, retrieval, and chat functionalities.
  - Pinecone: Used as the vector database to store and retrieve embeddings.

- **AI Models**:
  - `BAAI/bge-base-en-v1.5` for embedding generation.
  - `llama-3.3-70b-versatile` for language model responses.

- **Processing**:
  - PDF processing with `pdfplumber`.
  - DOCX processing with `python-docx`.

- **Containerization**:
  - Docker for containerized deployment.

---

## **Project Structure**
```
Trials                     # All trials
CHATBOT/
├── cvs_all/                   # Directory containing example resumes
├── src/                   # Source code directory
│   ├── app.py             # Gradio interface and chatbot logic
│   ├── file_processor.py  # Processes and embeds resumes
│   ├── rag_chain.py       # Handles retrieval and conversational AI logic
├── .env                   # Environment variables for API keys and secrets
├── .dockerignore          # Files and directories to exclude from Docker builds
├── Dockerfile             # Dockerfile for containerized deployment
├── docker-compose.yml     # Docker compose file
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```
---

## **Setup Instructions**

### Prerequisites
- Python 3.10.11
- pip 23.0.1
- Docker (for containerized deployment)

To structure your setup guide with three distinct options (local, Docker build, and Docker pull), here is a revised version of the instructions:

---

### Run the Chatbot: Three Options

#### Option 1: Run Locally
1. **Clone the Repository**  
   ```bash
   git clone <repository-url>
   cd CHATBOT
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys**  
   Set up `PINECONE_API_KEY` and `GROQ_API_KEY` as environment variables for embedding and chatbot functionality.

4. **Run the Application**  
   ```bash
   python src/app.py
   ```
   The application will launch a local Gradio interface.

---

#### Option 2: Run via Docker Build
1. **Build and Run the Docker Container**  
   ```bash
   docker build -t genai-chatbot .
   docker run -p 7860:7860 genai-chatbot
   ```

2. **Access the Chatbot**  
   Open [http://localhost:7860](http://localhost:7860) in your browser.

---

#### Option 3: Run via Docker Pull
1. **Pull the Pre-Built Docker Image**  
   ```bash
   docker pull fatmataha/chatbot
   ```

2. **Run the Container**  
   ```bash
   docker run -p 7860:7860 fatmataha/chatbot
   ```

3. **Access the Chatbot**  
   Open [http://localhost:7860](http://localhost:7860) in your browser.

---

This layout clearly separates the methods, allowing users to choose the most convenient option. Let me know if you'd like further adjustments!


---

## **Usage**

1. **Upload Resumes**:
   - Navigate to the "Upload Folder" tab.
   - Enter the directory path containing the resumes and upload them.

2. **Process Resumes**:
   - Switch to the "Process Files" tab and click "Process Files" to generate embeddings.
   - Optionally, clear the database by clicking "Clear Database."

3. **Chat with the Bot**:
   - Ask questions in the "Chat" tab to get recommendations for the best candidates based on your criteria.

---

## **Contributors**
- Team Name: M&M
- Team Members:
  - Mariam Elseedawy
  - Mariam Ashraf

---

## **License** 
[Apache License 2.0](LICENSE)
