import os
import PyPDF2
import docx
import re
import torch
import faiss
import streamlit as st
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

# Initialize transformer model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
transformer_model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def extract_text_from_pdf(pdf_file):
    with pdf_file as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text = re.sub(r"\.\.+", " ", text)
    return text

def extract_text_from_word(docx_file):
    word_reader = docx.Document(docx_file)
    text = ""
    for para in word_reader.paragraphs:
        text += para.text 
    text = re.sub(r"\.\.+", " ", text)
    return text

def extract_text_from_text(text_file):
    text = text_file.read().decode('latin-1')
    text = re.sub(r"\.\.+", " ", text)
    return text

def chunk_text(text, chunk_size=300):
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def get_embeddings(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = transformer_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

def process_uploaded_files(uploaded_files, chunk_size=200):
    index = faiss.IndexFlatL2(384)
    file_chunk_map = []
    
    for uploaded_file in uploaded_files:
        st.write(f"Processing {uploaded_file.name}...")
        
        if uploaded_file.name.endswith('.pdf'):
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith('.docx'):
            text = extract_text_from_word(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            text = extract_text_from_text(uploaded_file)
        else:
            st.write(f"Unsupported file type: {uploaded_file.name}")
            continue
            
        chunks = chunk_text(text, chunk_size=chunk_size)
        for chunk_id, chunk in enumerate(chunks):
            embedding = get_embeddings(chunk)
            index.add(embedding)
            file_chunk_map.append((uploaded_file.name, chunk_id, chunk))
            
    return index, file_chunk_map

def get_gemini_response(query, context):
    prompt = f"""Answer the following question using the provided context.   
                Question: {query}
                Context: {context}"""
    
    try:
        response = model.generate_content(prompt)
        
        if response.prompt_feedback.block_reason:
            return "Response was blocked due to content safety restrictions."
            
        return response.text.strip()
        
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit UI
st.title("Document Q&A System")
st.write("Upload documents and ask questions about their content!")

# File uploader
uploaded_files = st.file_uploader("Upload your documents", 
                                accept_multiple_files=True,
                                type=['pdf', 'docx', 'txt'])

if "index" not in st.session_state and uploaded_files:
    with st.spinner("Processing documents..."):
        st.session_state.index, st.session_state.file_chunk_map = process_uploaded_files(uploaded_files)
    st.success("Documents processed successfully!")

# Query input
query = st.text_input("Ask a question about your documents:")

if query and "index" in st.session_state:
    with st.spinner("Searching for answer..."):
        # Get query embedding and search
        query_embedding = get_embeddings(query)
        D, I = st.session_state.index.search(query_embedding, k=1)  # Get top 1 relevant chunks
        
        # Get relevant chunks
        results = []
        for i in range(len(I[0])):
            file_name, chunk_id, chunk = st.session_state.file_chunk_map[I[0][i]]
            results.append((file_name, chunk_id, chunk))
        
        # Combine chunks and get response
        context = " ".join([result[2] for result in results])
        response = get_gemini_response(query, context)
        
        # Display results
        st.write("### Answer:")
        st.write(response)
        
        with st.expander("View source chunks"):
            for file_name, chunk_id, chunk in results:
                st.write(f"**From {file_name} (Chunk {chunk_id}):**")
                st.write(chunk)
                st.write("---")

elif query:
    st.warning("Please upload some documents first!")

# Clear session state button
if st.button("Clear All"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("Session cleared! You can upload new documents.")