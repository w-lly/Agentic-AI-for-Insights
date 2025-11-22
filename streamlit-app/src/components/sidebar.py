"""
Sidebar Component
"""

import streamlit as st
from rag.pipeline_manager import initialize_rag_pipeline

def render_sidebar():
    """Render the sidebar with controls and information"""
    
    st.title("🎛️ Control Panel")
    
    # API Key input
    st.subheader("🔑 API Configuration")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=st.session_state.groq_api_key,
        help="Enter your Groq API key for LLM access"
    )
    
    if api_key:
        st.session_state.groq_api_key = api_key
        st.success("✅ API Key configured")
    else:
        st.warning("⚠️ Please enter your Groq API key")
    
    st.divider()
    
    # System initialization
    st.subheader("🚀 System Initialization")
    
    if st.button("Initialize RAG System", type="primary", disabled=not api_key):
        with st.spinner("Initializing RAG pipeline..."):
            try:
                initialize_rag_pipeline()
                st.session_state.system_initialized = True
                st.success("✅ RAG system initialized successfully!")
            except Exception as e:
                st.error(f"❌ Error initializing system: {str(e)}")
    
    # System status indicator
    if st.session_state.system_initialized:
        st.success("🟢 System Ready")
    else:
        st.info("🔵 System Not Initialized")
    
    st.divider()
    
    # Quick stats
    st.subheader("📊 Quick Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len(st.session_state.messages))
    with col2:
        st.metric("Documents", st.session_state.num_documents)
    
    st.divider()
    
    # Actions
    st.subheader("🔧 Actions")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔄 Reload Documents"):
        with st.spinner("Reloading documents..."):
            try:
                initialize_rag_pipeline()
                st.success("✅ Documents reloaded!")
            except Exception as e:
                st.error(f"❌ Error reloading: {str(e)}")
    
    st.divider()
    
    # Information
    with st.expander("ℹ️ About"):
        st.markdown("""
        **Coffee Machine RAG Assistant**
        
        This application uses Retrieval Augmented Generation (RAG) 
        to answer questions about coffee machine manuals.
        
        **Features:**
        - 📄 Multi-document support
        - 🔍 Hybrid retrieval (BM25 + Vector)
        - 🎯 Cross-encoder reranking
        - 🤖 LLM-powered responses
        
        **Supported Models:**
        - Groq (llama-3.3-70b-versatile)
        - Sentence Transformers for embeddings
        """)
    
    with st.expander("🛠️ Technical Details"):
        st.markdown(f"""
        **Current Configuration:**
        - Top-K: {st.session_state.top_k}
        - Chunk Size: {st.session_state.chunk_size} tokens
        - Chunk Overlap: {st.session_state.chunk_overlap} tokens
        - Temperature: {st.session_state.temperature}
        - Embedding Model: {st.session_state.embedding_model}
        """)