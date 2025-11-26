import streamlit as st
from pathlib import Path
import pandas as pd
from rag.document_processor import process_pdfs

def render_upload_tab():
    """Render document upload interface"""
    
    st.subheader("Upload PDF Documents")
    
    # Initialize upload counter in session state
    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF manual files",
        key=f"uploader_{st.session_state.upload_key}"
    )
    
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) selected")
        
        # Show file details
        file_df = pd.DataFrame({
            "Filename": [f.name for f in uploaded_files],
            "Size (KB)": [f.size / 1024 for f in uploaded_files]
        })
        st.dataframe(file_df, use_container_width=True)
    else:
        st.info("📁 No files selected. Please upload PDF documents to continue.")
    
    # Show existing documents with delete functionality
    st.divider()
    st.subheader("Existing Documents")
    has_files_to_delete = render_existing_documents()
    
    # Process button
    if uploaded_files or has_files_to_delete:
        if uploaded_files and has_files_to_delete:
            button_text = "Delete & Upload and Process Documents"
        elif has_files_to_delete:
            button_text = "Delete and Process Documents"
        else:
            button_text = "Upload and Process Documents"
            
        if st.button(button_text, type="primary"):
            process_documents_with_deletion(uploaded_files)
            st.rerun()

def render_existing_documents():
    """Display existing documents with delete functionality"""
    
    input_folder = Path("data/input")
    
    if not input_folder.exists():
        st.warning("⚠️ Input folder `data/input/` does not exist")
        return False
    
    existing_files = list(input_folder.glob("*.pdf"))
    
    if not existing_files:
        st.write("No documents found in `data/input/`")
        return False
    
    st.write(f"Found {len(existing_files)} existing document(s)")
    
    # Initialize session state for files to delete
    if "files_to_delete" not in st.session_state:
        st.session_state.files_to_delete = set()
    
    # Create checkboxes for each file
    for file in existing_files:
        file_size = file.stat().st_size / 1024
        is_checked = st.checkbox(
            f"{file.name} ({file_size:.1f} KB)", 
            key=f"delete_{file.name}",
            value=str(file) in st.session_state.files_to_delete
        )
        
        if is_checked:
            st.session_state.files_to_delete.add(str(file))
        else:
            st.session_state.files_to_delete.discard(str(file))
    
    # Show warning if files are marked for deletion
    if st.session_state.files_to_delete:
        st.warning(f"⚠️ {len(st.session_state.files_to_delete)} file(s) marked for deletion")
        return True
    
    return False

def process_documents_with_deletion(uploaded_files):
    """Delete marked files, upload new files, and reprocess all documents"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        input_folder = Path("data/input")
        input_folder.mkdir(parents=True, exist_ok=True)
        
        total_steps = len(st.session_state.files_to_delete) + (len(uploaded_files) if uploaded_files else 0) + 1
        current_step = 0
        
        # Delete marked files
        if st.session_state.files_to_delete:
            for file_path_str in st.session_state.files_to_delete:
                file_path = Path(file_path_str)
                if file_path.exists():
                    status_text.text(f"Deleting {file_path.name}...")
                    file_path.unlink()
                current_step += 1
                progress_bar.progress(current_step / total_steps)
            
            st.session_state.files_to_delete.clear()
        
        # Upload new files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                status_text.text(f"Uploading {uploaded_file.name}...")
                file_path = input_folder / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                current_step += 1
                progress_bar.progress(current_step / total_steps)
        
        # Process all PDFs
        status_text.text("Processing documents...")
        process_pdfs()
        
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        
        uploaded_count = len(uploaded_files) if uploaded_files else 0
        deleted_count = len(st.session_state.get('files_to_delete', []))
        
        if uploaded_count > 0 and deleted_count > 0:
            st.success(f"Uploaded {uploaded_count} file(s), deleted {deleted_count} file(s), and reprocessed documents")
        elif deleted_count > 0:
            st.success(f"Deleted {deleted_count} file(s) and reprocessed documents")
        else:
            st.success(f"Processed {uploaded_count} document(s)")
        
        st.balloons()
        
        # Increment upload key to reset file uploader
        st.session_state.upload_key += 1
        
    except Exception as e:
        st.error(f"❌ Error processing documents: {str(e)}")