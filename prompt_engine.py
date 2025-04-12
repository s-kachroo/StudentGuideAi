import os
import openai
from rag import load_pdf_documents, chunk_documents, get_vector_collection, query_cmu_knowledge


def get_chat_response(user_input):
    """
    Uses the RAG system (with the pre-built vector database)
    to generate an answer for the user query.
    """
    print(f"[INFO] User input: {user_input}")
    # Load the existing collection.
    collection = get_vector_collection(rebuild=False)
    print(f"[INFO] Collection loaded with {len(collection.get()['ids'])} chunks.")
    # Query the collection for relevant chunks.
    result = query_cmu_knowledge(collection, user_input)
    print(f"[INFO] Result: {result}")
    return result["answer"]
