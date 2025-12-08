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
from components.settings_manager import render_settings, initialize_settings
from utils.session_state import initialize_session_state

# Page configuration
st.set_page_config(
    page_title="7-11 Machine Assistant",
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
    
    # Initialize session state (only once)
    initialize_session_state()
    
    # Initialize settings from saved config
    initialize_settings()
    
    # Header
    st.markdown('<p class="main-header">☕ 7-11 Machine Assistant</p>', unsafe_allow_html=True)
    st.markdown("Ask questions about your machine manuals")
    
    # Sidebar
    with st.sidebar:
        render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["Chat", "Documents", "Settings"])
    
    with tab1:
        render_chat_interface()
    
    with tab2:
        render_document_manager()
    
    with tab3:
        render_settings()

if __name__ == "__main__":
    main()