import streamlit as st
from pathlib import Path
import pandas as pd

def render_view_tab():
    """Render document viewing interface"""
    
    st.subheader("View Processed Documents")
    
    output_file = Path("data/output/chunked_pdf_text.csv")
    
    if not output_file.exists():
        st.warning("⚠️ No processed documents found. Please upload and process documents first.")
        return
    
    try:
        df = pd.read_csv(output_file)
        st.session_state.num_chunks = len(df)
        
        st.info(f"Total chunks: {len(df)}")
        
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
        search_term = st.text_input("Search in chunks", "")
        
        if search_term:
            display_df = display_df[
                display_df['text'].str.contains(search_term, case=False, na=False)
            ]
            st.write(f"Found {len(display_df)} matching chunks")
        
        # Display chunks
        for idx, row in display_df.head(20).iterrows():
            with st.expander(f"{row['chunk_id']}"):
                st.text(row['text'])
                st.caption(f"Tokens: {row['start_token']} - {row['end_token']}")
        
        if len(display_df) > 20:
            st.info(f"Showing first 20 of {len(display_df)} chunks")
        
    except Exception as e:
        st.error(f"❌ Error loading documents: {str(e)}")