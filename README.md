# 📚 DocAI – Intelligent PDF Chat Assistant

DocAI is a powerful AI-powered Streamlit application that allows users to **upload PDF documents** (research papers, legal docs, health reports, etc.) and **interact with them conversationally** using natural language. It supports domain-specific responses and maintains session-based chat memory for meaningful interactions.

---

## 🚀 Features

- 📄 Upload and chat with **any PDF document**
- 🧠 Built using **LangChain**, **LLMs**, and **RAG architecture**
- 🩺 Supports domain-specific modes: Academic, Legal, Healthcare
- 🔁 Maintains **session-based chat memory** using `RunnableWithMessageHistory`
- 🔍 Uses **FAISS** vector store and **MiniLM embeddings**
- 📊 Includes **LangSmith tracing** for monitoring and debugging

---

## DEMO LINK: 
assets/DOCUAI.mp4

## 🛠️ Tech Stack

| Tool              | Purpose                                      |
|-------------------|----------------------------------------------|
| Streamlit         | Frontend UI                                  |
| LangChain         | LLM Orchestration                            |
| Groq / Llama3     | LLM backend (customizable)                   |
| HuggingFace       | Embeddings via `sentence-transformers`       |
| FAISS             | Vector store for retrieval                   |
| LangSmith         | Tracing and observability                    |
| PyMuPDF           | PDF parsing                                  |

---

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/docai-chat-assistant.git
cd docai-chat-assistant

2. **Install dependencies**
pip install -r requirements.txt

3. **Create a .env file**
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_key
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=your_project_name


Run the app:
streamlit run app.py


