import os
import re
import logging
from pathlib import Path
import PyPDF2
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import openai
from textwrap import dedent

# Set up logging
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)


def load_pdf_documents(data_dir="./data"):
    """
    Loads all PDF files from the specified directory.
    Returns a list of document dictionaries.
    """
    documents_list = []
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found at {data_path.absolute()}")
    pdf_files = list(data_path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_path.absolute()}")
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            text = ""
            with open(pdf_file, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    # Use a fallback in case extract_text() returns None
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
            documents_list.append({
                "filepath": str(pdf_file.absolute()),
                "filename": pdf_file.name,
                "title": pdf_file.stem,
                "text": text,
                "source": "CMU Official Documents"
            })
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {str(e)}")
            continue
    return documents_list


def clean_document_text(text):
    """
    Cleans text by removing extra spaces and some common unwanted patterns.
    """
    text = " ".join(text.split())
    patterns = [r"page \d+ of \d+", r"confidential", r"©\d+"]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text


def chunk_documents(documents_data, chunk_size=1000, chunk_overlap=200):
    """
    Splits each document's text into chunks (based on words).
    Optionally cleans the text before splitting.
    Returns a list of chunk dictionaries.
    """
    chunks_list = []
    for doc in tqdm(documents_data, desc="Chunking documents"):
        text = doc["text"]
        text = clean_document_text(text)
        words = text.split()
        # Slide a window over the word list
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_words = words[i: i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks_list.append({
                "text": chunk_text,
                "document_title": doc["title"],
                "document_source": doc["source"],
                "chunk_id": f"{doc['title']}_{len(chunk_text)}_{hash(chunk_text)}",
                "metadata": {
                    "source": doc["source"],
                    "title": doc["title"],
                    "filepath": doc["filepath"]
                }
            })
    return chunks_list


def get_vector_collection(rebuild=False, collection_name="cmu_student_guide", db_path="./chroma_db"):
    """
    Returns a persistent ChromaDB collection.
    If rebuild is set to True, PDFs are processed, chunked, and added to the collection;
    otherwise, the saved vector database (in ./chroma_db) is loaded.
    """
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )
    rag_collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
    )
    if rebuild:
        # Process PDFs and build the collection.
        documents = load_pdf_documents("./data")
        chunks = chunk_documents(documents)
        for i in tqdm(range(0, len(chunks), 100), desc="Indexing documents"):
            batch = chunks[i: i + 100]
            rag_collection.add(
                documents=[chunk["text"] for chunk in batch],
                metadatas=[chunk["metadata"] for chunk in batch],
                ids=[chunk["chunk_id"] for chunk in batch]
            )
    return rag_collection


def retrieve_relevant_chunks(rag_collection, query, top_k=3):
    """
    Queries the vector DB for the most similar document chunks to the query.
    Returns the raw query results.
    """
    results = rag_collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    # Optionally, add a similarity score if needed.
    results["scores"] = [1 - distance for distance in results["distances"][0]]
    return results


def generate_answer(query, retrieved_chunks, model="gpt-4o-mini"):
    """
    Generates an answer by constructing a prompt that includes the retrieved context
    and then calling the OpenAI ChatCompletion API.
    """
    context = "\n\n".join([
        f"Source: {meta['title']}\n{doc}"
        for doc, meta in zip(retrieved_chunks["documents"][0], retrieved_chunks["metadatas"][0])
    ])

    prompt = dedent(f"""
        You are a helpful CMU assistant. Answer based ONLY on this context:
        {context}
        Question: {query}
        Answer concisely and cite sources. If unsure, say you don't know.
        """)

    llm_response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a factual CMU student assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return llm_response.choices[0].message.content


def query_cmu_knowledge(rag_collection, user_question, top_k=3):
    """
    Retrieves document context using the vector DB and generates a final answer.
    Returns a dictionary with the question, answer, and source metadata.
    """
    try:
        retrieved_chunks = retrieve_relevant_chunks(rag_collection, user_question, top_k)
        answer = generate_answer(user_question, retrieved_chunks)
        return {
            "question": user_question,
            "answer": answer,
            "sources": retrieved_chunks["metadatas"][0]
        }
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        return {
            "question": user_question,
            "answer": "Sorry, I couldn't process your question. Please contact The HUB.",
            "sources": []
        }


if __name__ == "__main__":
    # Ensure the OpenAI API key is available.
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Build the vector database manually.
    print(f"[INFO] Rebuilding the vector database...")
    collection = get_vector_collection(rebuild=True)
    print(f"[INFO] Built the vector database with {len(collection.get()['ids'])} chunks.")

    # Run a sample queries to test the system.
    question1 = "What is the deadline to add a course?"
    response1 = query_cmu_knowledge(collection, question1)
    print(f"Question: {question1}, Answer: {response1['answer']}")
    question2 = "Who is Cathleen Kisak?"
    response2 = query_cmu_knowledge(collection, question2)
    print(f"Question: {question2}, Answer: {response2['answer']}")
