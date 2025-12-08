"""
Document Processing Module
"""

import os
from pathlib import Path
import pandas as pd
import pymupdf
import tiktoken
import streamlit as st

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    doc = pymupdf.open(pdf_path)
    extracted_text = ""
    
    for page in doc:
        extracted_text += page.get_text()
    
    return extracted_text

def chunk_text_with_metadata(text, chunk_size=None, overlap=None):
    """Chunk text with token-based splitting using user settings"""
    # Get settings from session state if available, otherwise use defaults
    if chunk_size is None:
        chunk_size = st.session_state.get('chunk_size', 500)
    if overlap is None:
        overlap = st.session_state.get('chunk_overlap', 50)
    
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)

        chunks.append({
            "chunk_id": chunk_id,
            "start_token": start,
            "end_token": end,
            "text": chunk_text
        })

        chunk_id += 1
        start += chunk_size - overlap
    
    return pd.DataFrame(chunks)

def chunk_multi_text_with_metadata(filenames, texts, chunk_size=None, overlap=None):
    """Chunk multiple texts with file-specific metadata using user settings"""
    assert len(filenames) == len(texts), "Length of filenames must match length of texts"
    
    # Get settings from session state if available, otherwise use defaults
    if chunk_size is None:
        chunk_size = st.session_state.get('chunk_size', 500)
    if overlap is None:
        overlap = st.session_state.get('chunk_overlap', 50)
    
    chunk_df_lst = []
    for i in range(len(texts)):
        chunk_df = chunk_text_with_metadata(texts[i], chunk_size, overlap)
        chunk_df["chunk_id"] = chunk_df["chunk_id"].apply(
            lambda x: f"{filenames[i]}_{x:04d}"
        )
        chunk_df_lst.append(chunk_df)
    
    return pd.concat(chunk_df_lst, ignore_index=True)

def process_pdfs():
    """Process all PDFs in the input folder using user settings"""
    
    input_folder = Path("data/input/")
    output_folder = Path("data/output/")
    
    # Create folders if they don't exist
    input_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all PDF files
    files = [f for f in os.listdir(input_folder) if f.endswith(".pdf")]
    
    if not files:
        raise Exception("No PDF files found in data/input/ folder")
    
    # Extract text from all PDFs
    df = pd.DataFrame(columns=["file_name", "text"])
    
    for file in files:
        path = input_folder / file
        extracted_text = extract_text_from_pdf(path)
        df.loc[df.shape[0]] = {"file_name": file, "text": extracted_text}
    
    # Save extracted text
    extracted_text_file = output_folder / "extracted_text_from_pdfs.csv"
    df.to_csv(extracted_text_file, index=False)
    
    # Get chunking settings from session state
    chunk_size = st.session_state.get('chunk_size', 500)
    chunk_overlap = st.session_state.get('chunk_overlap', 50)
    
    # Chunk the text with user-specified settings
    chunk_df = chunk_multi_text_with_metadata(
        df["file_name"], 
        df["text"],
        chunk_size=chunk_size,
        overlap=chunk_overlap
    )
    
    # Save chunked text
    chunked_text_file = output_folder / "chunked_pdf_text.csv"
    chunk_df.to_csv(chunked_text_file, index=False)
    
    return chunk_df