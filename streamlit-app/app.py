"""
Streamlit RAG Application - Main Entry Point
"""

import streamlit as st
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from components.sidebar import render_sidebar
from components.chat_interface import render_chat_interface
from components.document_manager import render_document_manager
from utils.session_state import initialize_session_state

# Page configuration
st.set_page_config(
    page_title="RAG Coffee Machine Assistant",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">☕ Coffee Machine RAG Assistant</p>', unsafe_allow_html=True)
    st.markdown("Ask questions about your coffee machine manuals")
    
    # Sidebar
    with st.sidebar:
        render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Documents", "⚙️ Settings"])
    
    with tab1:
        render_chat_interface()
    
    with tab2:
        render_document_manager()
    
    with tab3:
        render_settings()

def render_settings():
    """Render settings tab"""
    st.header("⚙️ System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Retrieval Settings")
        
        st.session_state.top_k = st.slider(
            "Number of documents to retrieve",
            min_value=1,
            max_value=10,
            value=st.session_state.get('top_k', 5),
            help="How many relevant documents to retrieve"
        )
        
        st.session_state.chunk_size = st.slider(
            "Chunk size (tokens)",
            min_value=100,
            max_value=1000,
            value=st.session_state.get('chunk_size', 500),
            help="Size of text chunks for processing"
        )
        
        st.session_state.chunk_overlap = st.slider(
            "Chunk overlap (tokens)",
            min_value=0,
            max_value=200,
            value=st.session_state.get('chunk_overlap', 50),
            help="Overlap between consecutive chunks"
        )
    
    with col2:
        st.subheader("Model Settings")
        
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('temperature', 0.0),
            step=0.1,
            help="Controls randomness in responses"
        )
        
        st.session_state.embedding_model = st.selectbox(
            "Embedding Model",
            options=[
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2"
            ],
            index=0,
            help="Model used for text embeddings"
        )
    
    st.divider()
    
    # System status
    st.subheader("📊 System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.metric("Documents Loaded", st.session_state.get('num_documents', 0))
    
    with status_col2:
        st.metric("Total Chunks", st.session_state.get('num_chunks', 0))
    
    with status_col3:
        vector_store_status = "✅ Ready" if st.session_state.get('vector_store') else "❌ Not Loaded"
        st.metric("Vector Store", vector_store_status)
    
    # Clear cache button
    if st.button("🗑️ Clear Cache & Reset", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()