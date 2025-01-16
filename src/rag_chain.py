import os
from langchain.embeddings import FastEmbedEmbeddings
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_pinecone import Pinecone
from langchain.memory import ConversationSummaryBufferMemory
import re

os.environ.get("GROQ_API_KEY", "PINECONE_API_KEY")
os.environ["GROQ_API_KEY"] =  "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"
os.environ["PINECONE_API_KEY"] = "pcsk_5WCgdB_QmWFymLpXWR9CFEaXsPggzWS83xBiBiLmDvWX8RMPdcT6G2x6Wa2p9LEQCvX

class RagChain:
    def __init__(self, index_name="new", embedding_model="BAAI/bge-base-en-v1.5", model_name="llama-3.3-70b-versatile", max_token_limit=1024):
        #self.folder_path = folder_path
        self.model_name = model_name
        self.groq_client = ChatGroq(temperature=0, model_name=model_name)
        self.embeddings = FastEmbedEmbeddings(model_name=embedding_model)

        # Initialize memory with summary buffer
        self.memory = ConversationSummaryBufferMemory(
            llm=self.groq_client,
            memory_key="chat_history",
            input_key="question",
            max_token_limit=max_token_limit,  # Controls the size of the buffer
            return_messages=True
        )

        # Initialize Pinecone vector store
        self.vector_db = Pinecone.from_existing_index(
            index_name=index_name,
            embedding=self.embeddings,
            text_key="text"
        )
        self.conversational_chain = self.create_qa_chain()

        # Track suggested candidates dynamically
        self.suggested_candidates = set()

    def create_qa_chain(self):
        """Creates a RetrievalQA chain with enhanced prompting and memory."""
        system_template = """
        You are an AI HR assistant. Your goal is to help users identify the best candidates for specific roles based on their requirements.
        Follow these rules:

        1. Use the given context about candidates to identify those who match the user's requirements (e.g., skills, experience).
        2. Rank candidates based on how well they meet the given criteria, explaining your reasoning.
        3. If the user asks for an alternative candidate, avoid suggesting candidates you have already mentioned.
        4. If no matching candidates are found, suggest the closest matches and explain why they could still be valuable.
        5. If the user's query is unclear or lacks details, ask specific follow-up questions to understand their needs better.
        6. If the user asks for candidates in an unrelated field (e.g., HR when context is technical), suggest switching focus or clarifying their request.
        7. When providing comparisons, highlight the strengths, weaknesses, and unique qualities of each candidate to help the user make informed decisions.
        8. Be concise and actionable, and avoid generic responses like "I don't have enough information." Instead, guide the user toward providing more relevant information.

        Context: {context}
        """
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")  # Key: "question"
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        return RetrievalQA.from_chain_type(
            llm=self.groq_client,
            retriever=self.vector_db.as_retriever(),
            memory=self.memory,
            chain_type="stuff",
            chain_type_kwargs={"prompt": chat_prompt},
            input_key="question"  # Explicitly set the input key to "question"
        )
    def ask_question(self, question):
        """Queries the chain dynamically, handles insufficient input, and utilizes memory."""
        try:
            # Dynamically exclude previously suggested candidates
            filtered_candidates = [{"not": {"person_name": name}} for name in self.suggested_candidates]
            retriever = self.vector_db.as_retriever(
                search_kwargs={"filter": {"$and": filtered_candidates}} if filtered_candidates else {}
            )

            self.conversational_chain.retriever = retriever

            # Pass the question using the correct key "question"
            response = self.conversational_chain.invoke({"question": question})

            # Handle insufficient or unclear responses
            result = response.get("result", "No result found.")
            if "I don't have enough information" in result or "No result found" in result:
                return (
                    "I couldn't find a perfect match for your query. Could you clarify or provide more specific details about "
                    "the skills, experience, or role you're looking for?"
                )

            # Extract candidate name dynamically from the response
            suggested_candidate = self.extract_candidate_from_response(result)
            if suggested_candidate:
                self.suggested_candidates.add(suggested_candidate)

            return result
        except Exception as e:
            print(f"Error while querying the chain: {e}")
            return "An error occurred while processing your question."

    def extract_candidate_from_response(self, response):
        """Extracts the suggested candidate's name from the response."""
        match = re.search(r"Candidate Name: ([\w\s]+)", response)
        if match:
            return match.group(1)
        return None

    def display_memory(self):
        """Displays the summarized memory."""
        try:
            memory_summary = self.memory.load_memory_variables({}).get("chat_history", "")
            if memory_summary:
                print("Memory Summary:")
                print(memory_summary)
            else:
                print("No memory has been summarized yet.")
        except Exception as e:
            print(f"Error while retrieving memory: {e}")
    
    def clear_memory(self):
        """Clears the conversation memory."""
        try:
            # Clear the internal memory buffer
            self.memory.chat_memory.clear()
            self.suggested_candidates.clear()
            print("Memory successfully cleared.")
        except Exception as e:
            print(f"Error while clearing memory: {e}")

    def reset_index(self):
        self.vector_db.delete_index(self.index_name)

        # Initialize Pinecone vector store
        self.vector_db = Pinecone.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings,
            text_key="text"
        )
