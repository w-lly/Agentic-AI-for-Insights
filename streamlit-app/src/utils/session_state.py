"""
Session State Management
"""

import streamlit as st

def initialize_session_state():
    """Initialize all session state variables"""
    
    # Chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # RAG components
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    if 'ensemble_retriever' not in st.session_state:
        st.session_state.ensemble_retriever = None
    
    if 'rag_app' not in st.session_state:
        st.session_state.rag_app = None
    
    # Document management
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    
    if 'num_documents' not in st.session_state:
        st.session_state.num_documents = 0
    
    if 'num_chunks' not in st.session_state:
        st.session_state.num_chunks = 0
    
    # Settings
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 5
    
    if 'chunk_size' not in st.session_state:
        st.session_state.chunk_size = 500
    
    if 'chunk_overlap' not in st.session_state:
        st.session_state.chunk_overlap = 50
    
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.0
    
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    
    # API keys
    if 'groq_api_key' not in st.session_state:
        st.session_state.groq_api_key = ""
    
    # Processing flags
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False

def reset_session_state():
    """Reset all session state variables"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()