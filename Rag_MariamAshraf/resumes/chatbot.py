# import gradio as gr
# from sentence_transformers import SentenceTransformer
# from pymilvus import Collection

# # Step 1: Connect to Milvus and load the collection
# collection = Collection("resume_collection")
# collection.load()

# # Step 2: Load the embedding model
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# # Step 3: Define the search function
# def search_resumes(query, k=3):
#     # Generate embedding for the query
#     query_embedding = embedding_model.encode(query)
#     query_embedding = [query_embedding.tolist()]  # Convert to list of lists

#     # Search the Milvus collection
#     results = collection.search(
#         query_embedding,
#         "embedding",
#         param={"metric_type": "L2", "params": {"nprobe": 10}},  # Search parameters
#         limit=k,  # Number of results to return
#         output_fields=["metadata"]  # Include metadata in the results
#     )

#     # Format the results
#     formatted_results = []
#     for hits in results:
#         for hit in hits:
#             formatted_results.append({
#                 "score": hit.score,
#                 "metadata": hit.entity.get("metadata"),
#                 "text": hit.entity.get("metadata")["source"]  # Use the source file as text
#             })
#     return formatted_results

# # Step 4: Define the chatbot interface
# def chatbot_interface(query, top_k=3):
#     results = search_resumes(query, top_k)
#     output = ""
#     for result in results:
#         output += f"Score: {result['score']}\n"
#         output += f"Source: {result['metadata']['source']}\n"
#         output += "-" * 50 + "\n"
#     return output

# # Step 5: Launch the Gradio app
# iface = gr.Interface(
#     fn=chatbot_interface,
#     inputs=["text", gr.Slider(1, 10, value=3, label="Number of Results")],
#     outputs="text",
#     title="Resume Screening Chatbot",
#     description="Ask questions about candidates' resumes."
# )
# iface.launch()










import gradio as gr
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

# Step 1: Load the FAISS index
index = faiss.read_index('resumes_index.faiss')

# Step 2: Load the chunks from chunks.pkl
with open('chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)

# Step 3: Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 4: Define the search function
def search_resumes(query, k=3):
    # Generate embedding for the query
    query_embedding = embedding_model.encode(query)
    query_embedding = np.array([query_embedding])

    # Search the FAISS index
    distances, indices = index.search(query_embedding, k)

    # Prepare the results
    results = []
    for i, idx in enumerate(indices[0]):
        result = {
            "rank": i + 1,
            "distance": float(distances[0][i]),
            "text": chunks[idx].page_content,
            "source": chunks[idx].metadata['source']
        }
        results.append(result)

    return results

# Step 5: Define the chatbot interface
def chatbot_interface(query):
    results = search_resumes(query)
    output = ""
    for result in results:
        output += f"Rank: {result['rank']}\n"
        output += f"Source: {result['source']}\n"
        output += f"Text: {result['text']}\n"
        output += "-" * 50 + "\n"
    return output

# Step 6: Launch the Gradio app
iface = gr.Interface(
    fn=chatbot_interface,
    inputs="text",
    outputs="text",
    title="Resume Screening Chatbot",
    description="Ask questions about candidates' resumes."
)
iface.launch()
