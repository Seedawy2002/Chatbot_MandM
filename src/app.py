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


def upload_folder(folder):
    """Handles folder upload."""
    global file_processor
    try:
        if folder:
            # Update file processor with the folder path
            file_processor = FileProcessor(
                folder_path=folder,
                index_name=index_name,
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            return "Folder uploaded successfully!"
        return "No folder uploaded."
    except Exception as e:
        return f"Error uploading folder: {e}"


def process_files():
    """Processes files in the uploaded folder."""
    global file_processor
    try:
        if file_processor:
            file_processor.process_files()
            return "Files processed and embeddings stored successfully!"
        return "No folder has been uploaded yet."
    except Exception as e:
        return f"Error processing files: {e}"


def reset_database():
    """Clears and recreates the vector database."""
    try:
        rag_chain.reset_index()
        return "Vector database cleared and recreated successfully!"
    except Exception as e:
        return f"Error resetting database: {e}"


def handle_user_input(chat_history, user_input):
    """Handles the user's input and appends it to the chat history."""
    if not user_input.strip():
        return chat_history, ""  # Return without changes if input is empty

    # Append user input to chat history
    chat_history.append({"role": "user", "content": user_input})
    return chat_history, ""


def generate_ai_response(chat_history):
    """Generates a response for the chatbot based on user input."""
    try:
        if chat_history and chat_history[-1]["role"] == "user":
            # Get the last user input
            user_input = chat_history[-1]["content"]

            # Generate a response from the RAG chain
            response = rag_chain.ask_question(user_input)

            # Prefix the response to clarify it's coming from the assistant
            if response:
                formatted_response = f"{response}"
                chat_history.append({"role": "assistant", "content": formatted_response})
            else:
                chat_history.append(
                    {"role": "assistant", "content": "I'm sorry, I couldn't process that. Could you provide more details?"}
                )
        return chat_history
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"Error generating response: {e}"})
        return chat_history


def clear_chat_history():
    """Clears the chat history and resets the chatbot."""
    try:
        rag_chain.clear_memory()  # Clear any memory in the RAG chain
    except Exception as e:
        return [{"role": "assistant", "content": f"Error clearing memory: {e}"}], []
    # Reset chat history
    return [{"role": "assistant", "content": "Hello! I'm your HR assistant. How can I help you find the best candidates?"}], []


def create_interface():
    """Creates the Gradio interface with three tabs."""

    with gr.Blocks() as demo:
        with gr.Tabs():
            # Tab 1: Upload Folder
            with gr.Tab("Upload Folder"):
                gr.Markdown("### Upload a Folder Containing Files")
                folder_input = gr.Textbox(label="Folder Path", placeholder="Enter the folder path")
                upload_button = gr.Button("Upload")
                upload_output = gr.Textbox(label="Upload Status", interactive=False)

                upload_button.click(upload_folder, inputs=[folder_input], outputs=[upload_output])

            # Tab 2: Process Files
            with gr.Tab("Process Files"):
                gr.Markdown("### Process Uploaded Files")
                process_button = gr.Button("Process Files")
                clear_button = gr.Button("Clear Database")  # New button
                process_output = gr.Textbox(label="Processing Status", interactive=False)

                process_button.click(process_files, inputs=[], outputs=[process_output])
                clear_button.click(reset_database, inputs=[], outputs=[process_output])  # New functionality

            # Tab 3: Chat
            with gr.Tab("Chat"):
                gr.Markdown("### Chat with RAG Chain")

                chat_history = gr.State([])  # Initialize with an empty chat history
                chatbot = gr.Chatbot(
                    label="Chatbot Interaction",
                    value=[{"role": "assistant", "content": "Hello! I'm your HR assistant. How can I help you find the best candidates?"}],  # Display the welcome message directly in the chat
                    type="messages"
                )

                user_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Type your question here",
                    lines=1,
                    interactive=True
                )

                send_button = gr.Button("Send")
                clear_button = gr.Button("Clear Chat History")

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
demo.launch()
