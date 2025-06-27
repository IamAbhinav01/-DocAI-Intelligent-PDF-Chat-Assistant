import os
from dotenv import load_dotenv
load_dotenv()

from io import BytesIO
import streamlit as st
from pathlib import Path
import tempfile

# Set environment variables
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

# --- PAGE CONFIG ---
st.set_page_config(
    layout="wide",
    page_title="DocAI",
    page_icon="📚"
)

# --- CUSTOM CSS ---
st.markdown("""
    <link href="https://fonts.googleapis.com/css?family=Poppins:400,600,700&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif !important;
        background-color: #f9fafb;
    }
    .main-header {
        background: linear-gradient(90deg, #1f2937 0%, #4b5563 100%);
        padding: 2.5rem 0 1.5rem 0;
        border-radius: 0 0 24px 24px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 24px rgba(31,41,55,0.08);
    }
    .main-header h1 {
        color: #fff;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: 1px;
    }
    .main-header h3 {
        color: #d1d5db;
        font-size: 1.25rem;
        font-weight: 400;
        margin-top: 0;
    }
    .main-header img {
        height: 56px;
        margin-bottom: 0.5rem;
    }
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #374151;
    }
    .chat-message {
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
        font-size: 16px;
        box-shadow: 0 2px 8px rgba(31,41,55,0.06);
        max-width: 90%;
        word-break: break-word;
    }
    .chat-message.user {
        background: #e0e7ef;
        color: #22223b;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    .chat-message.ai {
        background: #1f2937;
        color: #fff;
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }
    .chat-footer {
        position: sticky;
        bottom: 0;
        background: #f9fafb;
        padding-top: 1rem;
        z-index: 10;
    }
    .stChatInputContainer {
        padding-bottom: 1.5rem;
    }
    .stButton>button, .stDownloadButton>button {
        font-size: 14px !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
with st.container():
    st.markdown("""
        <div class="main-header">
            <img src="assets/DocuChatAI.png" alt="DocAI Logo" />
            <h1>DocAI: Intelligent PDF Chat Assistant</h1>
            <h3>Ask your research papers, legal documents, or health reports anything!</h3>
        </div>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<div class='sidebar-title'>📄 Upload your PDF</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-title'>Select Domain</div>", unsafe_allow_html=True)
    domain = st.selectbox("Select a domain", ["Academic/Research", "Legal", "Healthcare"], index=0, label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-title'>Session ID</div>", unsafe_allow_html=True)
    session_id = st.text_input("Enter Session ID")
    st.markdown("<br>", unsafe_allow_html=True)

# --- DOCUMENT PROCESSING ---
if uploaded_file:
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.prompts import MessagesPlaceholder
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.memory import ChatMessageHistory
    from langchain_core.runnables import RunnableWithMessageHistory

    import torch
    from sentence_transformers import SentenceTransformer

    # Save temp PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyMuPDFLoader(tmp_path, mode='page')
    pages = loader.load()

    # Split into chunks
    page_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    splited_pages = page_splitter.split_documents(pages)

    from langchain.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    texts = [doc.page_content for doc in splited_pages]
    _ = embeddings.embed_documents(texts)

    # Vectorstore
    vectorestoredb = FAISS.from_documents(splited_pages, embeddings)
    retriever = vectorestoredb.as_retriever()

    # LLM
    llm = ChatGroq(model="llama3-70b-8192")

    # Prompt
    domain_prompts = {
        "Academic/Research": "You are a research assistant. Answer in an academic tone with citations where applicable.",
        "Legal": "You are a legal expert. Reference legal clauses and explain in layman's terms.",
        "Healthcare": "You are a medical AI. Provide medically accurate responses and include disclaimers when needed."
    }
    system_prompt = domain_prompts.get(domain, "You are a helpful assistant. Use only the context below.")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, chain)

    # Session store for history
    session_store = {}
    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in session_store:
            session_store[session_id] = ChatMessageHistory()
        return session_store[session_id]

    with_message_history = RunnableWithMessageHistory(
        runnable=rag_chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

# --- MAIN CHAT UI ---
st.markdown("### Chat with your document", unsafe_allow_html=True)

# Session-level chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat Input
user_input = st.chat_input("Type your message and press Enter...")

if user_input:
    if not session_id:
        st.warning("Please enter a Session ID.")
    else:
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        config = {"configurable": {"session_id": session_id}}
        response = with_message_history.invoke({"input": user_input}, config=config)
        session_history = get_session_history(session_id)
        session_history.add_user_message(user_input)
        session_history.add_ai_message(response["answer"])
        ai_response = {"role": "ai", "message": response['answer']}
        st.session_state.chat_history.append(ai_response)
        st.rerun()

# Display chat
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["message"])

# Chat Controls
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🩹 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

with col2:
    chat_txt = "\n\n".join([f"{c['role'].capitalize()}: {c['message']}" for c in st.session_state.chat_history])
    st.download_button("⬇️ Download Chat (txt)", chat_txt, file_name="docai_chat.txt")
