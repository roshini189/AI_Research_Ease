import streamlit as st
import requests
import faiss
import numpy as np
import hashlib
import json
import os
import pdfplumber
from sentence_transformers import SentenceTransformer

# Function to generate a file hash
def generate_file_hash(file):
    file.seek(0)  # Ensure the file pointer is at the beginning
    return hashlib.sha256(file.read()).hexdigest()

# Function to extract text from PDF (Improved using pdfplumber)
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""  # Handle None case
    return text.strip()

# Cached function to load embedding model (avoiding multiple loads)
@st.cache_resource
def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

# Function to generate embeddings
def generate_embeddings(text):
    model = load_embedding_model()
    return model.encode(text, convert_to_numpy=True).astype('float32')

# Function to create or load FAISS index
def get_faiss_index():
    if os.path.exists("faiss_index.index"):
        return faiss.read_index("faiss_index.index")
    return None

# Function to save FAISS index
def save_faiss_index(index):
    faiss.write_index(index, "faiss_index.index")

# Function to load knowledge base
def load_knowledge_base():
    if os.path.exists("knowledge_base.json"):
        with open("knowledge_base.json", "r") as f:
            return json.load(f)
    return []

# Function to save knowledge base
def save_knowledge_base(knowledge_base):
    with open("knowledge_base.json", "w") as f:
        json.dump(knowledge_base, f)

# Function to clear the session state and delete saved files
def restart_session():
    st.session_state.clear()  # Clear the session state
    if os.path.exists("faiss_index.index"):
        os.remove("faiss_index.index")  # Delete the FAISS index file
    if os.path.exists("knowledge_base.json"):
        os.remove("knowledge_base.json")  # Delete the knowledge base file
    st.rerun()  # Rerun the app to reset the UI

# Function to retrieve relevant information
def retrieve_relevant_info(query_embedding, index, knowledge_base, top_k=3):
    if index is None or index.ntotal == 0:
        return ["No relevant documents found."]

    query_embedding = np.array([query_embedding]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    
    retrieved_docs = []
    for i in indices[0]:
        if 0 <= i < len(knowledge_base):
            retrieved_docs.append(knowledge_base[i])
    
    return retrieved_docs if retrieved_docs else ["No relevant documents found."]

def augment_prompt_with_retrieved_info(text, relevant_docs, query=None):
    """
    Modifies prompt structure based on whether it is an initial summarization 
    or a follow-up question.
    """
    retrieved_info = "\n\n".join(relevant_docs)

    if query:
        # If it's a user question, don't force a summary format
        augmented_prompt = f"Context Information:\n{retrieved_info}\n\nUser Question: {query}\n\nAnswer:"
    else:
        # Default summarization format
        augmented_prompt = f"Original Text:\n{text}\n\nRetrieved Information:\n{retrieved_info}\n\nSummary:"
    
    return augmented_prompt


# Function to call Ollama API for summarization or follow-up questions
def call_ollama_api(prompt, text=None):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "deepseek-r1:7b",
        "prompt": f"{prompt}: {text}" if text else prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, timeout=3000)  # Increase timeout to 30s
        response.raise_for_status()
        return response.json().get("response", "No response generated.")
    except requests.exceptions.RequestException as e:
        return f"Ollama API Error: {str(e)}"


# Streamlit UI
st.title("Research Paper Chatbot with RAG")
st.write("Upload a research paper (PDF) and chat with it!")

# Restart session button
if st.button("Restart Session"):
    restart_session()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = get_faiss_index()
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = load_knowledge_base()

# Ensure FAISS index is initialized
if st.session_state.faiss_index is None:
    st.session_state.faiss_index = faiss.IndexFlatL2(384)  # Default MiniLM embedding dimension

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf", key="file_uploader")

if uploaded_file is not None:
    file_hash = generate_file_hash(uploaded_file)
    
    if file_hash not in st.session_state.processed_files:
        st.session_state.processed_files.add(file_hash)

        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
            st.session_state.extracted_text = text
            st.session_state.knowledge_base.append(text)
            save_knowledge_base(st.session_state.knowledge_base)

        with st.spinner("Generating embeddings and setting up FAISS index..."):
            embeddings = generate_embeddings(text)
            embeddings = np.array([embeddings]).astype('float32')  # Ensure 2D shape for FAISS
            st.session_state.faiss_index.add(embeddings)
            save_faiss_index(st.session_state.faiss_index)

        with st.spinner("Generating summary with RAG..."):
            query_embedding = embeddings[0]
            relevant_docs = retrieve_relevant_info(query_embedding, st.session_state.faiss_index, st.session_state.knowledge_base, top_k=3)
            augmented_prompt = augment_prompt_with_retrieved_info(text, relevant_docs)
            summary = call_ollama_api(augmented_prompt)
            st.session_state.messages.append({"role": "assistant", "content": summary})
    
    else:
        st.warning("This file has already been processed.")

# Display extracted text preview
if "extracted_text" in st.session_state:
    with st.expander("Extracted Text Preview"):
        st.write(st.session_state.extracted_text[:1000] + "...")  # Show first 1000 characters

# Display FAISS status
if st.session_state.faiss_index.ntotal > 0:
    st.success(f"FAISS Index loaded with {st.session_state.faiss_index.ntotal} entries.")
else:
    st.warning("FAISS Index is empty. Upload a document to begin.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask a question about the research paper..."):
    if "extracted_text" not in st.session_state:
        st.error("Please upload a PDF file first.")
    else:
        # Add user's question to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate response to the question
        with st.spinner("Generating response..."):
            query_embedding = generate_embeddings(prompt)
            relevant_docs = retrieve_relevant_info(query_embedding, st.session_state.faiss_index, st.session_state.knowledge_base, top_k=3)
            
            # Pass the user question explicitly
            augmented_prompt = augment_prompt_with_retrieved_info(st.session_state.extracted_text, relevant_docs, query=prompt)
            response = call_ollama_api(prompt, augmented_prompt)

            st.session_state.messages.append({"role": "assistant", "content": response})

        # Display AI's response
        with st.chat_message("assistant"):
            st.write(response)

