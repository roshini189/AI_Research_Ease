# 🧠 AIResearchEase – Simplifying Research with AI 🤖📄

**AIResearchEase** is a secure, intelligent AI-powered web application designed to simplify academic research workflows. Built using Retrieval-Augmented Generation (RAG), FAISS-based semantic search, and local Large Language Models (LLMs), the app lets users upload research papers and ask context-aware questions, receiving instant, accurate answers.

---

## 🔍 Key Features

- 🧠 **AI-Powered Chatbot**  
  Ask questions about any uploaded research paper and get accurate, citation-aware responses using LLMs.

- 📄 **PDF Upload & Parsing**  
  Seamlessly extracts and chunks research paper content using `pdfplumber`.

- 📚 **Semantic Search with FAISS**  
  Embeds document sections using Sentence Transformers and retrieves the most relevant context.

- 💬 **Natural Language Q&A**  
  Interact using simple questions and get contextually grounded answers from the LLM.

- 🔐 **Secure & Local Execution**  
  Runs entirely on your machine using Ollama and Docker — no data leaves your environment.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python  
- **LLM Engine**: Ollama (e.g., Mistral, LLaMA2)  
- **Vector Search**: FAISS  
- **Embeddings**: Sentence Transformers  
- **PDF Parsing**: pdfplumber  
- **Containerization**: Docker (optional)

---

## 🔐 Security Highlights

- **Local-Only Inference** with Ollama (no external API calls)  
- **AES-level Secure File Handling**  

