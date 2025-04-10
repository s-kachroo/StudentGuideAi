import os, openai
from openai import OpenAI

# Setup OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai.api_key)

def get_chat_response(user_input, model_name="gpt-4o-mini"):
    prompt = f"You are a helpful assistant. Answer this query: {user_input}"
    try:
        response = client.chat.completions.create(
            model=model_name,
            store=True,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"[ERROR] Error processing the prompt: {e}"
    print(f"Prompt: {prompt}")
    print(f"Response: {answer}")
    return answer

# # Import required libraries
# import os
# import re
# import PyPDF2
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# import chromadb
# from chromadb.utils import embedding_functions
# from openai import OpenAI
# import json
# from bert_score import score as bert_score
# import logging
# from pathlib import Path
#
# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Silence non-critical errors from PyPDF
# logging.getLogger("PyPDF2").setLevel(logging.CRITICAL)
