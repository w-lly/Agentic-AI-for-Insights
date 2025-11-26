import streamlit as st
from pathlib import Path
import pandas as pd

def render_statistics_tab():
    """Render document statistics"""
    
    st.subheader("Document Statistics")
    
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