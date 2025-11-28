"""
Settings Manager Component
"""

import streamlit as st
import json
from pathlib import Path

# Configuration file paths
CONFIG_FILE = Path("data/config/user_settings.json")
PROCESSING_METADATA_FILE = Path("data/config/processing_metadata.json")

DEFAULT_SETTINGS = {
    'top_k': 5,
    'chunk_size': 500,
    'chunk_overlap': 50,
    'temperature': 0.0,
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
}

EMBEDDING_OPTIONS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2"
]

# Settings that require document reprocessing
REPROCESSING_SETTINGS = {'chunk_size', 'chunk_overlap', 'embedding_model'}

# Settings that require RAG pipeline reinitialization
REINIT_SETTINGS = {'top_k', 'temperature'}


def load_settings():
    """Load user settings from config file"""
    settings = DEFAULT_SETTINGS.copy()
    
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                saved_settings = json.load(f)
                settings.update(saved_settings)
    except Exception:
        pass
    
    return settings


def save_settings(settings):
    """Save user settings to config file"""
    try:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        st.error(f"Could not save settings: {e}")


def load_processing_metadata():
    """Load metadata about how documents were last processed"""
    try:
        if PROCESSING_METADATA_FILE.exists():
            with open(PROCESSING_METADATA_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return None


def save_processing_metadata(settings):
    """Save metadata about document processing settings"""
    try:
        PROCESSING_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        metadata = {key: settings[key] for key in REPROCESSING_SETTINGS}
        with open(PROCESSING_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        st.error(f"Could not save processing metadata: {e}")


def check_reprocessing_needed():
    """Check if documents need reprocessing based on settings"""
    current_settings = load_settings()
    last_processing = load_processing_metadata()
    
    if last_processing is None:
        return False
    
    return any(
        last_processing.get(key) != current_settings[key] 
        for key in REPROCESSING_SETTINGS
    )


def get_changed_settings(old_settings, new_settings):
    """Determine which types of settings changed"""
    reprocessing_needed = any(
        old_settings[key] != new_settings[key] 
        for key in REPROCESSING_SETTINGS
    )
    
    reinit_needed = any(
        old_settings[key] != new_settings[key] 
        for key in REINIT_SETTINGS
    )
    
    return reprocessing_needed, reinit_needed


def reinitialize_pipeline():
    """Reinitialize pipeline components for runtime settings changes"""
    try:
        from rag.pipeline_manager import reinitialize_retrievers_and_llm
        
        reinitialize_retrievers_and_llm()
        
        # Track what the pipeline is now using
        for key in REINIT_SETTINGS:
            st.session_state[f'pipeline_{key}'] = st.session_state[key]
        
        return True
    except Exception as e:
        raise Exception(f"Failed to reinitialize pipeline: {str(e)}")


def reprocess_documents():
    """Reprocess documents with current settings"""
    from rag.document_processor import process_pdfs
    
    # Process documents
    process_pdfs()
    
    # Save processing metadata
    current_settings = {key: st.session_state[key] for key in REPROCESSING_SETTINGS}
    save_processing_metadata(current_settings)
    
    # Clear reprocessing flag
    st.session_state.reprocessing_needed = False


def initialize_settings():
    """Initialize settings in session state from saved config"""
    saved_settings = load_settings()
    
    for key, value in saved_settings.items():
            st.session_state[key] = value
    
    # Check if reprocessing is needed on startup
    if 'reprocessing_needed' not in st.session_state:
        st.session_state.reprocessing_needed = check_reprocessing_needed()


def render_settings():
    """Render settings tab"""
    st.header("System Settings")
    
    col1, col2 = st.columns(2)
    
    # Get current values
    current_settings = {key: st.session_state.get(key, DEFAULT_SETTINGS[key]) for key in DEFAULT_SETTINGS}
    
    # Settings UI
    with col1:
        st.subheader("Retrieval Settings")
        
        new_top_k = st.slider(
            "Number of documents to retrieve (Top K)",
            min_value=1,
            max_value=10,
            value=current_settings['top_k'],
            help="How many relevant document chunks to retrieve and rerank"
        )
        
        new_chunk_size = st.slider(
            "Chunk size (tokens)",
            min_value=100,
            max_value=1000,
            value=current_settings['chunk_size'],
            help="Size of text chunks for processing"
        )
        
        new_chunk_overlap = st.slider(
            "Chunk overlap (tokens)",
            min_value=0,
            max_value=200,
            value=current_settings['chunk_overlap'],
            help="Overlap between consecutive chunks"
        )
    
    with col2:
        st.subheader("Model Settings")
        
        new_temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=current_settings['temperature'],
            step=0.1,
            help="Controls randomness in responses"
        )
        
        current_index = EMBEDDING_OPTIONS.index(current_settings['embedding_model']) \
                       if current_settings['embedding_model'] in EMBEDDING_OPTIONS else 0
        
        new_embedding_model = st.selectbox(
            "Embedding Model",
            options=EMBEDDING_OPTIONS,
            index=current_index,
            help="Model used for text embeddings and retrieval"
        )
    
    new_settings = {
        'top_k': new_top_k,
        'chunk_size': new_chunk_size,
        'chunk_overlap': new_chunk_overlap,
        'temperature': new_temperature,
        'embedding_model': new_embedding_model
    }
    
    # Check what changed
    settings_changed = any(current_settings[key] != new_settings[key] for key in DEFAULT_SETTINGS)
    reprocessing_needed_on_save, reinit_needed_on_save = get_changed_settings(current_settings, new_settings)
    
    # Save and Reset buttons
    col_save, col_reset = st.columns([1, 1])
    
    with col_save:
        if st.button("Save Settings", type="primary", use_container_width=True, disabled=not settings_changed):
            # Update session state
            for key, value in new_settings.items():
                st.session_state[key] = value
            
            save_settings(new_settings)
            
            # Handle reprocessing settings changes
            if reprocessing_needed_on_save:
                st.session_state.reprocessing_needed = True
                st.success("✅ Settings saved successfully!")
                st.warning("⚠️ Document processing settings changed. Please reprocess documents below.")
            
            # Handle runtime settings changes (only if RAG already initialized and no reprocessing needed)
            elif reinit_needed_on_save and st.session_state.get('rag_app'):
                try:
                    reinitialize_pipeline()
                    st.success("✅ Settings saved and pipeline updated successfully!")
                except Exception as e:
                    st.error(f"❌ Settings saved but failed to update pipeline: {str(e)}")
            else:
                st.success("✅ Settings saved successfully!")
            
            st.rerun()
    
    with col_reset:
        if st.button("Reset to Defaults", type="secondary", use_container_width=True):
            # Update session state
            for key, value in DEFAULT_SETTINGS.items():
                st.session_state[key] = value
            
            save_settings(DEFAULT_SETTINGS)
            
            # Check what will change
            reprocessing_will_be_needed, reinit_will_be_needed = get_changed_settings(
                current_settings, DEFAULT_SETTINGS
            )
            
            if reprocessing_will_be_needed:
                st.session_state.reprocessing_needed = True
                st.success("✅ Settings reset to defaults!")
                st.warning("⚠️ Document processing settings changed. Please reprocess documents below.")
            elif reinit_will_be_needed and st.session_state.get('rag_app'):
                try:
                    reinitialize_pipeline()
                    st.success("✅ Settings reset and pipeline updated successfully!")
                except Exception as e:
                    st.error(f"❌ Settings reset but failed to update pipeline: {str(e)}")
            else:
                st.success("✅ Settings reset to defaults!")
            
            st.rerun()
    
    # Reprocess documents section
    if st.session_state.get('reprocessing_needed', False):
        st.divider()
        st.warning("⚠️ **Document reprocessing required** - Current settings don't match how documents were last processed.")
        
        col_reprocess, col_ignore = st.columns([2, 1])
        
        with col_reprocess:
            if st.button("Reprocess Documents Now", type="primary", use_container_width=True):
                try:
                    with st.spinner("Reprocessing documents with current settings..."):
                        reprocess_documents()
                        st.success("✅ Documents reprocessed successfully!")
                        
                        # Set flag to show reinit button if RAG was already initialized
                        if st.session_state.get('rag_app'):
                            st.session_state.reinit_after_reprocess = True
                        
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error reprocessing documents: {str(e)}")
        
        with col_ignore:
            if st.button("Dismiss", type="secondary", use_container_width=True):
                st.session_state.reprocessing_needed = False
                st.rerun()
    
    # Reinitialize RAG button (shown after reprocessing if RAG was previously initialized)
    if st.session_state.get('reinit_after_reprocess', False):
        st.divider()
        st.info("ℹ️ **RAG pipeline reinitialization available** - Documents have been reprocessed. Reinitialize the pipeline to use the new settings.")
        
        if st.button("Reinitialize RAG Pipeline", type="primary", use_container_width=True):
            try:
                with st.spinner("Reinitializing RAG pipeline..."):
                    from rag.pipeline_manager import initialize_rag_pipeline
                    initialize_rag_pipeline()
                    st.session_state.reinit_after_reprocess = False
                    st.success("✅ RAG pipeline reinitialized successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error reinitializing pipeline: {str(e)}")
    
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
    
    # Settings comparison
    st.divider()
    st.subheader("Settings Comparison")
    
    col_current, col_processed = st.columns(2)
    
    with col_current:
        st.write("**Current Active Settings:**")
        st.json(current_settings)
    
    with col_processed:
        st.write("**Last Processing Settings:**")
        last_processing = load_processing_metadata()
        if last_processing:
            st.json(last_processing)
        else:
            st.info("No documents processed yet")