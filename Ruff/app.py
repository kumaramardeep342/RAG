from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz
import re
import requests
from transformers import AutoTokenizer, AutoModel
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def load_chunk_embeddings(file_path):
    embedding_matrix = np.load(file_path)
    print(f"Chunk embeddings loaded from {file_path}.")
    return embedding_matrix
def load_faiss_index(file_path):
    index = faiss.read_index(file_path)
    print(f"FAISS index loaded from {file_path}.")
    return index
def load_chunks_from_file(file_path):
    with open(file_path, "r") as f:
        chunks = [line.strip() for line in f]
    print(f"Chunks loaded from {file_path}.")
    return chunks
def retrieve_relevant_chunks(query, faiss_index, chunk_texts, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    model = AutoModel.from_pretrained(embedding_model_name)

    # Compute embedding for the query
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    query_embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()

    # Search in FAISS index
    _, indices = faiss_index.search(query_embedding, k=5)  # Retrieve top 5 chunks
    return [chunk_texts[i] for i in indices[0]]


API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
API_TOKEN = os.getenv('ACCESS_TOKEN')

def query_huggingface_api(prompt):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.5,
            "top_p":0.9,
            "repetition_penalty":1.2
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error: {response.status_code}, {response.text}"


embedding_matrix = load_chunk_embeddings("data/chunk_embeddings.npy")
faiss_index = load_faiss_index("data/faiss_index.bin")
chunk_texts = load_chunks_from_file("data/chunks.txt")
def generate_response(query):
    retrieved_chunks = retrieve_relevant_chunks(query, faiss_index, chunk_texts)
    context = "\n".join(retrieved_chunks)
    prompt = f"Context: {context}\n\nQuery: {query}\n\nAnswer:"
    outputs = query_huggingface_api(prompt)
    print(outputs)
    if 'Answer:' in outputs:
        answer=outputs.split('Answer:')[-1].strip()
        return answer
    else:
        return outputs

st.title("Food Recipes & Safety Precautions")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("I'm all ears, fire away!")

if query:
    answer = generate_response(query)  # Generate answer
    st.session_state.history.append({"query": query, "answer": answer})

# Display conversation history
for qa in st.session_state.history:
    st.write(f"**Q:** {qa['query']}")
    st.write(f"**A:** {qa['answer']}")

