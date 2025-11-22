"""
Document Manager Component
"""

import streamlit as st
import os
from pathlib import Path
import pandas as pd
from rag.document_processor import process_pdfs, extract_text_from_pdf

def render_document_manager():
    """Render the document management interface"""
    
    st.header("📄 Document Management")
    
    tab1, tab2, tab3 = st.tabs(["Upload Documents", "View Documents", "Document Statistics"])
    
    with tab1:
        render_upload_tab()
    
    with tab2:
        render_view_tab()
    
    with tab3:
        render_statistics_tab()

def render_upload_tab():
    """Render document upload interface"""
    
    st.subheader("📤 Upload PDF Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF manual files"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) selected")
        
        # Show file details
        file_df = pd.DataFrame({
            "Filename": [f.name for f in uploaded_files],
            "Size (KB)": [f.size / 1024 for f in uploaded_files]
        })
        st.dataframe(file_df, use_container_width=True)
        
        if st.button("Process Documents", type="primary"):
            process_uploaded_documents(uploaded_files)
    else:
        st.info("📁 No files selected. Please upload PDF documents to continue.")
        
        # Show existing documents location
        st.subheader("📂 Document Location")
        input_folder = Path("data/input")
        if input_folder.exists():
            existing_files = list(input_folder.glob("*.pdf"))
            if existing_files:
                st.write(f"Found {len(existing_files)} existing document(s) in `data/input/`:")
                for file in existing_files:
                    st.write(f"- {file.name}")
            else:
                st.write("No documents found in `data/input/`")
        else:
            st.warning("⚠️ Input folder `data/input/` does not exist")

def process_uploaded_documents(uploaded_files):
    """Process uploaded PDF documents"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Create input directory if it doesn't exist
        input_folder = Path("data/input")
        input_folder.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded files
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Saving {uploaded_file.name}...")
            
            file_path = input_folder / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            progress_bar.progress((i + 1) / (len(uploaded_files) + 1))
        
        status_text.text("Processing documents...")
        
        # Process PDFs
        from rag.pipeline_manager import initialize_rag_pipeline
        initialize_rag_pipeline()
        
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        
        st.success(f"Successfully processed {len(uploaded_files)} document(s)")
        st.balloons()
        
    except Exception as e:
        st.error(f"❌ Error processing documents: {str(e)}")

def render_view_tab():
    """Render document viewing interface"""
    
    st.subheader("📑 View Processed Documents")
    
    # Check for processed data
    output_file = Path("data/output/chunked_pdf_text.csv")
    
    if not output_file.exists():
        st.warning("⚠️ No processed documents found. Please upload and process documents first.")
        return
    
    try:
        # Load chunked data
        df = pd.read_csv(output_file)
        st.session_state.num_chunks = len(df)
        
        st.info(f"📊 Total chunks: {len(df)}")
        
        # File filter
        unique_files = df['chunk_id'].str.extract(r'^(.+?)_\d+$')[0].unique()
        selected_file = st.selectbox(
            "Select document to view",
            options=["All"] + list(unique_files)
        )
        
        # Filter data
        if selected_file != "All":
            display_df = df[df['chunk_id'].str.startswith(selected_file)]
        else:
            display_df = df
        
        st.write(f"Showing {len(display_df)} chunks")
        
        # Search functionality
        search_term = st.text_input("🔍 Search in chunks", "")
        
        if search_term:
            display_df = display_df[
                display_df['text'].str.contains(search_term, case=False, na=False)
            ]
            st.write(f"Found {len(display_df)} matching chunks")
        
        # Display chunks
        for idx, row in display_df.head(20).iterrows():
            with st.expander(f"📄 {row['chunk_id']}"):
                st.text(row['text'])
                st.caption(f"Tokens: {row['start_token']} - {row['end_token']}")
        
        if len(display_df) > 20:
            st.info(f"Showing first 20 of {len(display_df)} chunks")
        
    except Exception as e:
        st.error(f"❌ Error loading documents: {str(e)}")

def render_statistics_tab():
    """Render document statistics"""
    
    st.subheader("📊 Document Statistics")
    
    output_file = Path("data/output/chunked_pdf_text.csv")
    
    if not output_file.exists():
        st.warning("⚠️ No processed documents found.")
        return
    
    try:
        df = pd.read_csv(output_file)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Chunks", len(df))
        
        with col2:
            unique_docs = df['chunk_id'].str.extract(r'^(.+?)_\d+$')[0].nunique()
            st.metric("Source Documents", unique_docs)
        
        with col3:
            total_tokens = (df['end_token'] - df['start_token']).sum()
            st.metric("Total Tokens", f"{total_tokens:,}")
        
        st.divider()
        
        # Chunks per document
        st.subheader("Chunks per Document")
        
        df['document'] = df['chunk_id'].str.extract(r'^(.+?)_\d+$')[0]
        chunks_per_doc = df['document'].value_counts()
        
        st.bar_chart(chunks_per_doc)
        
        st.divider()
        
        # Token distribution
        st.subheader("Token Distribution")
        
        df['chunk_length'] = df['end_token'] - df['start_token']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{df['chunk_length'].mean():.0f}")
        with col2:
            st.metric("Median", f"{df['chunk_length'].median():.0f}")
        with col3:
            st.metric("Min", f"{df['chunk_length'].min():.0f}")
        with col4:
            st.metric("Max", f"{df['chunk_length'].max():.0f}")
        
        st.line_chart(df['chunk_length'])
        
    except Exception as e:
        st.error(f"❌ Error loading statistics: {str(e)}")