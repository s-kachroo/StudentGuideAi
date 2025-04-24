import os
import openai
import pickle
from rag import (
    load_pdf_documents,
    chunk_documents,
    create_default_vector_collection,
    create_question_vector_collection,
    query_cmu_knowledge
)
from config import (
    OPENAI_API_KEY,
    DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    DB_PATH_DEFAULT, DB_PATH_QINDEX,
    TOP_K, NUM_QUESTIONS, CHUNKS_CACHE_PATH
)

# == CONFIG ==
openai.api_key = OPENAI_API_KEY
USE_QUESTION_INDEX = True  # True to use the question-index pipeline, False to use the original chunk-based pipeline.

# == PRELOAD CHUNKS ==
if os.path.exists(CHUNKS_CACHE_PATH):
    with open(CHUNKS_CACHE_PATH, "rb") as f:
        _chunks = pickle.load(f)
    print(f"[INFO] Loaded {len(_chunks)} chunks from cache.")
else:
    _documents = load_pdf_documents(DATA_DIR)
    _chunks    = chunk_documents(_documents, CHUNK_SIZE, CHUNK_OVERLAP)
    with open(CHUNKS_CACHE_PATH, "wb") as f:
        pickle.dump(_chunks, f)
    print(f"[INFO] Chunked {len(_chunks)} docs and cached to {CHUNKS_CACHE_PATH}.")

# == PRELOAD VECTOR COLLECTIONS ==
_collection_default  = create_default_vector_collection(build=False, chunks=_chunks, collection_name="cmu_student_guide", db_path=DB_PATH_DEFAULT)
print(f"[INFO] Default RAG loaded {len(_collection_default.get()['ids'])} chunks.")
_collection_questions = create_question_vector_collection(build=False, chunks=_chunks, collection_name="cmu_question_index", db_path=DB_PATH_QINDEX, num_questions=NUM_QUESTIONS)
print(f"[INFO] Question‐index RAG loaded {len(_collection_questions.get()['ids'])} chunks.")

def get_chat_response(user_input):
    """
    Uses the RAG system (with the pre-built vector database)
    to generate an answer for the user query.
    """
    print(f"[INFO] User input: {user_input}")
        
    # Query the collection for the answer to the user question.
    if USE_QUESTION_INDEX:
        # route through the question-index pipeline
        resp = query_cmu_knowledge(
            user_question = user_input,
            use_questions  = True,
            d_collection   = None,
            q_collection   = _collection_questions,
            chunks         = _chunks,
            top_k          = TOP_K
        )
    else:
        # route through the original chunk-based pipeline
        resp = query_cmu_knowledge(
            user_question = user_input,
            use_questions  = False,
            d_collection   = _collection_default,
            q_collection   = None,
            chunks         = None,
            top_k          = TOP_K
        )    
    
    print(f"[INFO] Answer: {resp['answer']}")
    return resp["answer"]    
