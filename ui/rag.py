import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import pickle
import logging
from pathlib import Path
import pandas as pd
import PyPDF2
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import openai
from openai import OpenAI
from textwrap import dedent
from torchmetrics.text.bert import BERTScore
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")
from config import (
    OPENAI_API_KEY,
    DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    DB_PATH_DEFAULT, DB_PATH_QINDEX,
    TOP_K, NUM_QUESTIONS, CHUNKS_CACHE_PATH
)

# ——— Setup —————————————————————————————————————————————————————————
openai.api_key = OPENAI_API_KEY
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# ——— 1) PDF Loading + Chunking (with cache) ———————————————————————————

def load_pdf_documents(data_dir):
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


def chunk_documents(documents_data, chunk_size, chunk_overlap):
    """
    Splits each document's text into chunks (based on words).
    Optionally cleans the text before splitting.
    Returns a list of chunk dictionaries.
    """
    chunks_list = []
    for doc in tqdm(documents_data, desc="Chunking documents"):
        text = doc["text"]
        text = clean_document_text(text=text)
        words = text.split()
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


def load_or_cache_chunks(data_dir, chunk_size, chunk_overlap):
    """
    Loads and chunks PDFs once, then caches to disk.
    On subsequent runs loads directly from cache.
    """
    if os.path.exists(CHUNKS_CACHE_PATH):
        with open(CHUNKS_CACHE_PATH, "rb") as f:
            chunks = pickle.load(f)
        print(f"[INFO] Loaded {len(chunks)} chunks from cache.")
    else:
        docs = load_pdf_documents(data_dir)
        chunks = chunk_documents(docs, chunk_size, chunk_overlap)
        with open(CHUNKS_CACHE_PATH, "wb") as f:
            pickle.dump(chunks, f)
        print(f"[INFO] Chunked {len(chunks)} docs and wrote cache.")
    return chunks


# ——— 2) Question Generation ——————————————————————————————


def generate_questions_for_chunk(chunk, num_questions, model_name):
    """
    Given one chunk dict (with keys 'text' and 'chunk_id'),
    generate `num_questions` that can be answered from it.
    Returns a list of question strings.
    """
    prompt = dedent(f"""
        You are given the following excerpt from a CMU policy document:
        \"\"\"{chunk['text']}\"\"\"

        Generate {num_questions} clear, concise questions that can be answered
        using ONLY this excerpt. Return them as:
        Q1: ...
        Q2: ...
        Q3: ...
    """).strip()

    client = OpenAI(api_key=openai.api_key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    questions = []
    for line in resp.choices[0].message.content.splitlines():
        m = re.match(r"Q\d+:\s*(.+)", line.strip())
        if m:
            questions.append(m.group(1).strip())
    time.sleep(0.05)
    return questions


# ——— 3) Vector DB Builders —————————————————————————————————————————


def create_default_vector_collection(build=False, chunks=None, collection_name="cmu_student_guide", db_path="./chroma_db"):
    """
    Returns a persistent ChromaDB collection.
    If build is set to True, PDFs are processed, chunked, and added to the collection;
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
    if build:
        for i in tqdm(range(0, len(chunks), 100), desc="Indexing documents"):
            batch = chunks[i: i + 100]
            rag_collection.add(
                documents=[chunk["text"] for chunk in batch],
                metadatas=[chunk["metadata"] for chunk in batch],
                ids=[chunk["chunk_id"] for chunk in batch]
            )
    return rag_collection


def create_question_vector_collection(build=False, chunks=None, collection_name="cmu_question_index", db_path="./chroma_qdb", num_questions=5):
    """
    Build or load a ChromaDB collection whose 'documents' are
    LL-generated questions, each tagged with chunk_id metadata.
    """
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )
    qcol = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
    )
    if build:
        for chunk in tqdm(chunks, desc="Generating & indexing questions"):
            qs = generate_questions_for_chunk(chunk=chunk, num_questions=num_questions, model_name="gpt-4o-mini")
            for idx, q in enumerate(qs):
                qid = f"{chunk['chunk_id']}_q{idx}"
                qcol.add(
                    documents=[q],
                    metadatas=[{"chunk_id": chunk["chunk_id"]}],
                    ids=[qid]
                )
    return qcol


# ——— 4) Retrieval & Answering ————————————————————————————————————————


def retrieve_relevant_chunks(rag_collection, query, top_k):
    """
    Queries the vector DB for the most similar document chunks to the query.
    Returns the raw query results.
    """
    results = rag_collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    results["scores"] = [1 - distance for distance in results["distances"][0]]
    return results


def retrieve_relevant_chunks_via_questions(qcol, chunks, query, top_k):
    """
    1) Query the question index for the top_k most similar LL-generated questions.
    2) Collect the unique chunk_ids from their metadata.
    3) Return the corresponding chunk dicts.
    """
    res = qcol.query(
        query_texts=[query],
        n_results=top_k,
        include=["metadatas"]
    )
    ids = {m["chunk_id"] for m in res["metadatas"][0]}
    return [c for c in chunks if c["chunk_id"] in ids]


def generate_answer(query, retrieved_chunks, model_name="gpt-4o-mini"):
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
        Please give concise answer with less than 15 words, meaningful, and cite sources, if possible.
        """)

    client = OpenAI(api_key=openai.api_key)
    llm_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a factual CMU student assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return llm_response.choices[0].message.content


def query_cmu_knowledge(user_question, use_questions=False, d_collection=None, q_collection=None, chunks=None, top_k=3):
    """
    Retrieves document context using the vector DB and generates a final answer.
    Returns a dictionary with the question, answer, and source metadata.
    """    
    try:
        if use_questions:
            assert q_collection and chunks, "need question_col and chunks for use_questions=True"
            selected_chunks = retrieve_relevant_chunks_via_questions(qcol=q_collection, chunks=chunks, query=user_question, top_k=top_k)
            retrieved = {"documents": [[c["text"] for c in selected_chunks]], "metadatas": [[c["metadata"] for c in selected_chunks]]}
        else:
            retrieved = retrieve_relevant_chunks(rag_collection=d_collection, query=user_question, top_k=top_k)

        answer = generate_answer(query=user_question, retrieved_chunks=retrieved)
        return {
            "question": user_question,
            "answer": answer,
            "sources": retrieved["metadatas"][0]
        }
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return {
            "question": user_question,
            "answer": "Sorry, I couldn't process your question. Please contact The HUB.",
            "sources": []
        }


if __name__ == "__main__":
    # Load or cache your chunks
    chunks = load_or_cache_chunks(DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP)

    # Build the default vector database.
    print(f"\n[INFO] Building the default vector database...")
    collection_default = create_default_vector_collection(build=True, chunks=chunks, collection_name="cmu_student_guide", db_path=DB_PATH_DEFAULT)
    print(f"[INFO] Built the vector database with {len(collection_default.get()['ids'])} chunks.")

    # Build the question index vector database.
    print(f"\n[INFO] Building the question index vector database...")
    collection_question = create_question_vector_collection(build=True, chunks=chunks, collection_name="cmu_question_index", db_path=DB_PATH_QINDEX, num_questions=NUM_QUESTIONS)
    print(f"[INFO] Built the question index with {len(collection_question.get()['ids'])} questions.")

    # Run a sample queries to test the system.
    question1 = "What is the deadline to add a course?"
    response1 = query_cmu_knowledge(user_question=question1, use_questions=False, d_collection=collection_default, chunks=chunks, top_k=TOP_K)
    print(f"\nQuestion: {question1}, Answer: {response1['answer']}")
    question2 = "Who is Cathleen Kisak?"
    response2 = query_cmu_knowledge(user_question=question2, use_questions=True, q_collection=collection_question, chunks=chunks, top_k=TOP_K)
    print(f"\nQuestion: {question2}, Answer: {response2['answer']}")
