import gradio as gr
from file_processor import FileProcessor
from rag_chain import RagChain
import os

# Initialize global variables
uploaded_folder_path = "uploaded_folder"

# Ensure the folder exists
os.makedirs(uploaded_folder_path, exist_ok=True)

# Initialize RagChain and FileProcessor
index_name = "new"  # Specify your Pinecone index name
embedding_model = "BAAI/bge-base-en-v1.5"
chunk_size = 1000
chunk_overlap = 100

rag_chain = RagChain(index_name=index_name, embedding_model=embedding_model)
file_processor = None
processing = False  # Track if an AI response is being processed

def upload_folder(folder):
    """Handles folder upload."""
    global file_processor
    if folder:
        # Update file processor with the folder path
        file_processor = FileProcessor(folder_path=folder, index_name=index_name, embedding_model=embedding_model,
                                        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return "Folder uploaded successfully!"
    return "No folder uploaded."

def process_files():
    """Processes files in the uploaded folder."""
    global file_processor
    if file_processor:
        file_processor.process_files()
        return "Files processed and embeddings stored successfully!"
    return "No folder has been uploaded yet."

def reset_database():
    """Clears and recreates the vector database."""
    try:
        rag_chain.reset_index()
        return "Vector database cleared and recreated successfully!"
    except Exception as e:
        return f"Error resetting database: {e}"

def create_interface():
    """Creates the Gradio interface with three tabs."""

    with gr.Blocks() as demo:
        with gr.Tabs():
            # Tab 1: Upload Folder
            with gr.Tab("Upload Folder"):
                gr.Markdown("### Upload a Folder Containing Files")
                folder_input = gr.Textbox(label="Folder Path", placeholder="Enter the folder path")
                upload_button = gr.Button("Upload")
                upload_output = gr.Textbox(label="Upload Status")

                upload_button.click(upload_folder, inputs=[folder_input], outputs=[upload_output])

            # Tab 2: Process Files
            with gr.Tab("Process Files"):
                gr.Markdown("### Process Uploaded Files")
                process_button = gr.Button("Process Files")
                clear_button = gr.Button("Clear Database")  # New button
                process_output = gr.Textbox(label="Processing Status")

                process_button.click(process_files, inputs=[], outputs=[process_output])
                clear_button.click(reset_database, inputs=[], outputs=[process_output])  # New functionality

            # Tab 3: Chat
            with gr.Tab("Chat"):
                gr.Markdown("### Chat with RAG Chain")

                chat_history = gr.State([{"role": "assistant", "content": "Hello! I'm your HR assistant. How can I help you find the best candidate or prepare for interviews today?"}])  # Initialize with a welcome message
                chatbot = gr.Chatbot(
                    label="Chatbot Interaction",
                    value=[{"role": "assistant", "content": "Hello! I'm your HR assistant. How can I help you find the best candidates?"}],
                    type="messages"
                )

                user_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Type your question here",
                    lines=1,
                    interactive=True  # Make it clear the input field is editable
                )

                send_button = gr.Button("Send")  # Send button for user convenience

                clear_button = gr.Button("Clear Chat History")  # Clear chat history button

                def handle_user_input(chat_history, user_input):
                    global processing
                    if processing or not user_input.strip():
                        return chat_history, ""  # Do nothing if processing or input is empty

                    chat_history.append({"role": "user", "content": user_input})
                    return chat_history, ""

                def generate_ai_response(chat_history):
                    global processing
                    if chat_history:
                        user_input = chat_history[-1]["content"]
                        response = rag_chain.ask_question(user_input)
                        chat_history.append({"role": "assistant", "content": response})
                    processing = False
                    return chat_history

                def clear_chat_history():
                    global rag_chain
                    rag_chain.clear_memory()
                    return [{"role": "assistant", "content": "Hello! I'm your HR assistant. How can I help you find the best candidates?"}], []

                send_button.click(
                    handle_user_input,
                    inputs=[chat_history, user_input],
                    outputs=[chatbot, user_input]
                )

                send_button.click(
                    generate_ai_response,
                    inputs=[chat_history],
                    outputs=[chatbot],
                    queue=True
                )

                user_input.submit(
                    handle_user_input,
                    inputs=[chat_history, user_input],
                    outputs=[chatbot, user_input]
                )

                user_input.submit(
                    generate_ai_response,
                    inputs=[chat_history],
                    outputs=[chatbot],
                    queue=True
                )

                clear_button.click(
                    clear_chat_history,
                    inputs=[],
                    outputs=[chatbot, chat_history]
                )

    return demo

# Run the Gradio app
demo = create_interface()
demo.launch(server_name="0.0.0.0", server_port=7860)
