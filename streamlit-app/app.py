"""
Streamlit RAG Application - Main Entry Point
"""

import streamlit as st
from pathlib import Path
import sys
import json

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from components.sidebar import render_sidebar
from components.chat_interface import render_chat_interface
from components.document_manager import render_document_manager
from utils.session_state import initialize_session_state

# Configuration file path
CONFIG_FILE = Path("data/config/user_settings.json")

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

def load_settings():
    """Load user settings from config file"""
    default_settings = {
        'top_k': 5,
        'chunk_size': 500,
        'chunk_overlap': 50,
        'temperature': 0.0,
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
    }
    
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                saved_settings = json.load(f)
                # Merge with defaults to handle new settings
                default_settings.update(saved_settings)
    except Exception as e:
        st.warning(f"Could not load settings: {e}")
    
    return default_settings

def save_settings(settings):
    """Save user settings to config file"""
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        st.error(f"Could not save settings: {e}")

def initialize_settings():
    """Initialize settings in session state from saved config"""
    # Force reload settings on every run to ensure persistence
    saved_settings = load_settings()
    
    # Always update from saved settings to ensure persistence across refreshes
    for key, value in saved_settings.items():
        st.session_state[key] = value

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize settings from saved config
    initialize_settings()
    
    # Header
    st.markdown('<p class="main-header">☕ Coffee Machine RAG Assistant</p>', unsafe_allow_html=True)
    st.markdown("Ask questions about your coffee machine manuals")
    
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

def render_settings():
    """Render settings tab"""
    st.header("⚙️ System Settings")
    
    # Track if settings have changed
    if 'settings_changed' not in st.session_state:
        st.session_state.settings_changed = False
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Retrieval Settings")
        
        new_top_k = st.slider(
            "Number of documents to retrieve",
            min_value=1,
            max_value=10,
            value=st.session_state.get('top_k', 5),
            key='slider_top_k',
            help="How many relevant documents to retrieve"
        )
        
        new_chunk_size = st.slider(
            "Chunk size (tokens)",
            min_value=100,
            max_value=1000,
            value=st.session_state.get('chunk_size', 500),
            key='slider_chunk_size',
            help="Size of text chunks for processing"
        )
        
        new_chunk_overlap = st.slider(
            "Chunk overlap (tokens)",
            min_value=0,
            max_value=200,
            value=st.session_state.get('chunk_overlap', 50),
            key='slider_chunk_overlap',
            help="Overlap between consecutive chunks"
        )
    
    with col2:
        st.subheader("Model Settings")
        
        new_temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('temperature', 0.0),
            step=0.1,
            key='slider_temperature',
            help="Controls randomness in responses"
        )
        
        new_embedding_model = st.selectbox(
            "Embedding Model",
            options=[
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2"
            ],
            index=0 if st.session_state.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2') == 'sentence-transformers/all-MiniLM-L6-v2' else 1,
            key='select_embedding_model',
            help="Model used for text embeddings"
        )
    
    # Check if settings have changed from saved values
    settings_changed = (
        new_top_k != st.session_state.get('top_k', 5) or
        new_chunk_size != st.session_state.get('chunk_size', 500) or
        new_chunk_overlap != st.session_state.get('chunk_overlap', 50) or
        new_temperature != st.session_state.get('temperature', 0.0) or
        new_embedding_model != st.session_state.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
    )
    
    # Check if chunking or embedding settings changed (requires reprocessing)
    reprocessing_needed = (
        new_chunk_size != st.session_state.get('chunk_size', 500) or
        new_chunk_overlap != st.session_state.get('chunk_overlap', 50) or
        new_embedding_model != st.session_state.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
    )
    
    # Save settings button
    col_save, col_reset = st.columns([1, 1])
    
    with col_save:
        if st.button("💾 Save Settings", type="primary", use_container_width=True, disabled=not settings_changed):
            # Update session state
            st.session_state.top_k = new_top_k
            st.session_state.chunk_size = new_chunk_size
            st.session_state.chunk_overlap = new_chunk_overlap
            st.session_state.temperature = new_temperature
            st.session_state.embedding_model = new_embedding_model
            
            # Save to file
            settings_to_save = {
                'top_k': new_top_k,
                'chunk_size': new_chunk_size,
                'chunk_overlap': new_chunk_overlap,
                'temperature': new_temperature,
                'embedding_model': new_embedding_model
            }
            save_settings(settings_to_save)
            
            # Mark if reprocessing is needed
            if reprocessing_needed:
                st.session_state.settings_changed = True
            
            st.success("✅ Settings saved successfully!")
            if reprocessing_needed:
                st.warning("⚠️ Chunking or embedding settings changed. Please reprocess documents below.")
            st.rerun()
    
    with col_reset:
        if st.button("🔄 Reset to Defaults", type="secondary", use_container_width=True):
            default_settings = {
                'top_k': 5,
                'chunk_size': 500,
                'chunk_overlap': 50,
                'temperature': 0.0,
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
            }
            
            # Update session state
            for key, value in default_settings.items():
                st.session_state[key] = value
            
            # Save to file
            save_settings(default_settings)
            st.session_state.settings_changed = False
            
            st.success("✅ Settings reset to defaults!")
            st.rerun()
    
    # Reprocess documents button (only show if settings changed)
    if st.session_state.get('settings_changed', False):
        st.divider()
        st.warning("⚠️ Document reprocessing required due to chunking or embedding settings change.")
        
        if st.button("🔄 Reprocess Documents", type="primary", use_container_width=True):
            try:
                with st.spinner("Reprocessing documents with new settings..."):
                    # Import the function
                    from rag.document_processor import process_pdfs
                    
                    # Call the reprocessing function
                    process_pdfs()
                    
                    # Clear the flag
                    st.session_state.settings_changed = False
                    
                    st.success("✅ Documents reprocessed successfully with new settings!")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error reprocessing documents: {str(e)}")
                st.exception(e)
    
    st.divider()
    
    # System status
    st.subheader("System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.metric("Documents Loaded", st.session_state.get('num_documents', 0))
    
    with status_col2:
        st.metric("Total Chunks", st.session_state.get('num_chunks', 0))
    
    with status_col3:
        vector_store_status = "✅ Ready" if st.session_state.get('vector_store') else "❌ Not Loaded"
        st.metric("Vector Store", vector_store_status)
    
    # Current settings display
    st.divider()
    st.subheader("Current Settings")
    
    settings_display = {
        "Top K": st.session_state.get('top_k', 5),
        "Chunk Size": st.session_state.get('chunk_size', 500),
        "Chunk Overlap": st.session_state.get('chunk_overlap', 50),
        "Temperature": st.session_state.get('temperature', 0.0),
        "Embedding Model": st.session_state.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
    }
    
    st.json(settings_display)
    
    # Clear cache button
    st.divider()
    if st.button("🗑️ Clear Cache & Reset Session", type="secondary"):
        # Keep settings but clear other session state
        settings_to_keep = {
            'top_k': st.session_state.get('top_k'),
            'chunk_size': st.session_state.get('chunk_size'),
            'chunk_overlap': st.session_state.get('chunk_overlap'),
            'temperature': st.session_state.get('temperature'),
            'embedding_model': st.session_state.get('embedding_model')
        }
        
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Restore settings
        for key, value in settings_to_keep.items():
            if value is not None:
                st.session_state[key] = value
        
        st.rerun()

if __name__ == "__main__":
    main()