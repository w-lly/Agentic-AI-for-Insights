"""
RAG Pipeline Manager
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import Literal
import faiss

from .document_processor import process_pdfs


def save_faiss_vector_store(vector_store, folder_path: str):
    """Save a LangChain FAISS vector_store to folder_path"""
    vector_store.save_local(folder_path)
    print(f"FAISS vector store saved to '{folder_path}'.")


def load_faiss_vector_store(folder_path: str, model_name: str):
    """Load a LangChain FAISS vector store from folder_path with embedding function"""
    embedding_function = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.load_local(
        folder_path,
        embedding_function,
        allow_dangerous_deserialization=True
    )
    print(f"FAISS vector store loaded from '{folder_path}'.")
    return vector_store


def create_faiss_vector_store(df, docs, embedding_model):
    """Create FAISS vector store from documents and embeddings"""
    # Ensure chunk_id is unique
    assert df["chunk_id"].is_unique, "chunk_id must be unique"
    
    # Create embeddings
    model = SentenceTransformer(embedding_model)
    texts = df["text"].tolist()
    embeddings = model.encode(texts, convert_to_numpy=True, batch_size=32)
    embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Get embedding dimension
    embedding_dim = embeddings_normalized.shape[1]
    
    # Create FAISS index (IP for cosine similarity)
    index = faiss.IndexFlatIP(embedding_dim)
    
    # Add embeddings to FAISS index (must be float32)
    index.add(embeddings_normalized.astype("float32"))
    
    # Create embedding function
    embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # Create FAISS vector store in memory (RAM)
    vector_store = FAISS(
        embedding_function=embedding_function,
        index=index,
        docstore=InMemoryDocstore({chunk_id: doc for chunk_id, doc in zip(df["chunk_id"], docs)}),
        index_to_docstore_id={i: chunk_id for i, chunk_id in enumerate(df["chunk_id"])}
    )
    
    return vector_store


def initialize_rag_pipeline():
    """Initialize the complete RAG pipeline"""
    
    try:
        # Step 1: Process PDFs if needed
        output_file = Path("data/output/chunked_pdf_text.csv")
        
        if not output_file.exists():
            process_pdfs()

        # Step 2: Load chunked data
        df = pd.read_csv(output_file)
        st.session_state.num_chunks = len(df)
        st.session_state.num_documents = df['chunk_id'].str.extract(r'^(.+?)_\d+$')[0].nunique()
        
        # Step 3: Create documents
        texts = df["text"].tolist()
        docs = [Document(page_content=text) for text in texts]
        st.session_state.documents = docs
        
        # Step 4: Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = st.session_state.top_k
        
        # Step 5: Load or create FAISS vector store
        faiss_db_path = Path("data/faiss_database")
        
        if faiss_db_path.exists():
            vector_store = load_faiss_vector_store(
                str(faiss_db_path),
                st.session_state.embedding_model
            )
        else:
            # Create new FAISS index
            vector_store = create_faiss_vector_store(df, docs, st.session_state.embedding_model)
            
            # Save FAISS database
            faiss_db_path.mkdir(parents=True, exist_ok=True)
            save_faiss_vector_store(vector_store, str(faiss_db_path))
        
        st.session_state.vector_store = vector_store
        
        # Step 6: Create FAISS retriever
        faiss_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": st.session_state.top_k}
        )
        
        # Step 7: Create ensemble retriever (hybrid retriever combining embeddings and keyword search)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )
        
        # Step 8: Create reranker
        cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        reranker_compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)
        
        # Create contextual compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker_compressor,
            base_retriever=ensemble_retriever
        )
        
        st.session_state.ensemble_retriever = ensemble_retriever
        st.session_state.compression_retriever = compression_retriever
        
        # Step 9: Initialize LLM
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=st.session_state.temperature,
            groq_api_key=st.session_state.groq_api_key
        )
        
        st.session_state.llm = llm
        
        # Step 10: Create retriever tool
        retriever_tool = create_retriever_tool(
            ensemble_retriever,
            name="ensemble_retriever_tool",
            description="Search user manuals and technical documents."
        )
        
        # Step 11: Build LangGraph workflow
        workflow = build_langgraph_workflow(llm, retriever_tool, compression_retriever)
        
        st.session_state.rag_app = workflow
        
        # Track what settings the pipeline was initialized with
        st.session_state.pipeline_top_k = st.session_state.top_k
        st.session_state.pipeline_temperature = st.session_state.temperature
        
        return True
        
    except Exception as e:
        print(e)
        raise Exception(f"Failed to initialize RAG pipeline: {str(e)}")

def reinitialize_retrievers_and_llm():
    """
    Reinitialize retrievers and LLM with updated top_k and temperature settings.
    This is called when only top_k or temperature changes (no full reprocessing needed).
    """
    
    if not st.session_state.get('documents'):
        raise Exception("No documents loaded. Please initialize the pipeline first.")
    
    try:
        docs = st.session_state.documents
        
        # Update BM25 retriever with new top_k
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = st.session_state.top_k
        
        # Update FAISS retriever with new top_k
        vector_store = st.session_state.vector_store
        faiss_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": st.session_state.top_k}
        )
        
        # Recreate ensemble retriever with updated retrievers
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )
        
        # Recreate reranker
        cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        reranker_compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)
        
        # Recreate contextual compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker_compressor,
            base_retriever=ensemble_retriever
        )
        
        st.session_state.ensemble_retriever = ensemble_retriever
        st.session_state.compression_retriever = compression_retriever
        
        # Reinitialize LLM with updated temperature
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=st.session_state.temperature,
            groq_api_key=st.session_state.groq_api_key
        )
        
        st.session_state.llm = llm
        
        # Recreate retriever tool
        retriever_tool = create_retriever_tool(
            ensemble_retriever,
            name="ensemble_retriever_tool",
            description="Search user manuals and technical documents."
        )
        
        # Rebuild LangGraph workflow with updated components
        workflow = build_langgraph_workflow(llm, retriever_tool, compression_retriever)
        
        st.session_state.rag_app = workflow
        
        print(f"Pipeline reinitialized with top_k={st.session_state.top_k}, temperature={st.session_state.temperature}")
        
    except Exception as e:
        raise Exception(f"Failed to reinitialize pipeline components: {str(e)}")

def build_langgraph_workflow(llm, retriever_tool, compression_retriever):
    """Build the LangGraph workflow with reranking and grading"""
    
    # Define grading schema
    class GradeDocuments(BaseModel):
        binary_score: str = Field(description="yes or no")
    
    # Prompts
    GRADE_PROMPT = (
        "You are grading whether the retrieved context is relevant to the user's question.\n"
        "Question: {question}\n"
        "Context: {context}\n\n"
        "Is this context relevant? Reply 'yes' or 'no'."
    )
    
    SYNTH_PROMPT = (
        "You are an assistant. Use the retrieved context to answer the question.\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n\n"
        "Provide a concise and accurate answer."
    )
    
    def generate_query_or_respond(state: MessagesState):
        """Generate query or respond directly"""
        response = llm.bind_tools([retriever_tool]).invoke(state["messages"])
        return {"messages": [response]}
    
    def rerank_documents(state: MessagesState):
        """Rerank retrieved documents using cross-encoder"""
        question = state["messages"][0].content
        
        # Retrieve compressed documents using reranker
        compressed_docs = compression_retriever.invoke(question)
        
        # Combine top docs into a single string
        reranked_context = "\n\n".join(doc.page_content for doc in compressed_docs)
        
        # Append as a proper message
        state["messages"].append(HumanMessage(content=reranked_context))
        return {"messages": state["messages"]}
    
    def grade_documents(state: MessagesState) -> Literal["generate_answer", END]:
        """Grade whether documents are relevant"""
        question = state["messages"][0].content
        context = state["messages"][-1].content
        
        prompt = GRADE_PROMPT.format(question=question, context=context)
        response = llm.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
        
        return "generate_answer" if response.binary_score.lower() == "yes" else END
    
    def generate_answer(state: MessagesState):
        """Generate final answer using synthesizer"""
        question = state["messages"][0].content
        context = state["messages"][-1].content
        
        prompt = SYNTH_PROMPT.format(context=context, question=question)
        response = llm.invoke([{"role": "user", "content": prompt}])
        
        return {"messages": [response]}
    
    # Create a new StateGraph
    workflow = StateGraph(MessagesState)
    
    # Add the nodes to the graph
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rerank_documents", rerank_documents)
    workflow.add_node(generate_answer)
    
    # Define the edges (the flow)
    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", END: END}
    )
    workflow.add_edge("retrieve", "rerank_documents")
    workflow.add_conditional_edges("rerank_documents", grade_documents)
    workflow.add_edge("generate_answer", END)
    
    print("LangGraph workflow compiled successfully.")
    return workflow.compile()


def query_rag_system(query: str):
    """Query the RAG system and return response with contexts"""
    
    if not st.session_state.rag_app:
        raise Exception("RAG system not initialized. Please initialize first.")
    
    # Run the workflow
    result = st.session_state.rag_app.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    
    # Extract response
    response = result["messages"][-1].content
    
    # Get contexts from compression retriever
    contexts = []
    try:
        docs = st.session_state.compression_retriever.invoke(query)
        contexts = [doc.page_content for doc in docs]
    except Exception as e:
        print(f"Context retrieval error: {e}")
        contexts = ["Context retrieval failed"]
    
    return response, contexts