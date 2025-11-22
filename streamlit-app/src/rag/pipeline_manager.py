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
import faiss

from .document_processor import process_pdfs

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
        
        # Step 5: Create embeddings
        embedding_function = HuggingFaceEmbeddings(
            model_name=st.session_state.embedding_model
        )
        
        # Step 6: Load or create FAISS vector store
        faiss_db_path = Path("data/faiss_database")
        
        if faiss_db_path.exists():
            vector_store = FAISS.load_local(
                str(faiss_db_path),
                embedding_function,
                allow_dangerous_deserialization=True
            )
        else:
            # Create new FAISS index
            model = SentenceTransformer(st.session_state.embedding_model)
            embeddings = model.encode(texts, convert_to_numpy=True, batch_size=32)
            embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            embedding_dim = embeddings_normalized.shape[1]
            index = faiss.IndexFlatIP(embedding_dim)
            index.add(embeddings_normalized.astype("float32"))
            
            vector_store = FAISS(
                embedding_function=embedding_function,
                index=index,
                docstore=InMemoryDocstore({df.iloc[i]['chunk_id']: doc for i, doc in enumerate(docs)}),
                index_to_docstore_id={i: df.iloc[i]['chunk_id'] for i in range(len(docs))}
            )
            
            # Save FAISS database
            faiss_db_path.mkdir(parents=True, exist_ok=True)
            vector_store.save_local(str(faiss_db_path))
        
        st.session_state.vector_store = vector_store
        
        # Step 7: Create FAISS retriever
        faiss_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": st.session_state.top_k}
        )
        
        # Step 8: Create ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )
        
        # Step 9: Create reranker
        cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        reranker_compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker_compressor,
            base_retriever=ensemble_retriever
        )
        
        st.session_state.ensemble_retriever = compression_retriever
        
        # Step 10: Initialize LLM
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=st.session_state.temperature,
            groq_api_key=st.session_state.groq_api_key
        )
        
        # Step 11: Create retriever tool
        retriever_tool = create_retriever_tool(
            compression_retriever,
            name="manual_retriever",
            description="Search coffee machine user manuals for relevant information."
        )
        
        # Step 12: Build LangGraph workflow
        workflow = build_langgraph_workflow(llm, retriever_tool)
        
        st.session_state.rag_app = workflow
        
        return True
        
    except Exception as e:
        raise Exception(f"Failed to initialize RAG pipeline: {str(e)}")

def build_langgraph_workflow(llm, retriever_tool):
    """Build the LangGraph workflow"""
    
    def generate_query(state: MessagesState):
        response = llm.bind_tools([retriever_tool]).invoke(state["messages"])
        return {"messages": [response]}
    
    def rerank_documents(state: MessagesState):
        # Extract contexts from tool messages
        contexts = []
        for msg in state["messages"]:
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                contexts.append(msg.content)
        return {"messages": state["messages"]}
    
    def generate_answer(state: MessagesState):
        question = state["messages"][0].content
        
        # Extract context from messages
        context = ""
        for msg in state["messages"][1:]:
            if hasattr(msg, 'content'):
                context += str(msg.content) + "\n\n"
        
        prompt = f"""You are an assistant for coffee machine manuals. Use the retrieved context to answer questions accurately.

Context:
{context}

Question: {question}

Provide a clear, concise, and accurate answer based on the context."""
        
        response = llm.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}
    
    # Build workflow
    workflow = StateGraph(MessagesState)
    
    workflow.add_node("generate_query", generate_query)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("generate_answer", generate_answer)
    
    workflow.add_edge(START, "generate_query")
    workflow.add_conditional_edges("generate_query", tools_condition, {"tools": "retrieve", END: END})
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
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
    
    # Get contexts from retriever
    contexts = []
    try:
        docs = st.session_state.ensemble_retriever.invoke(query)
        contexts = [doc.page_content for doc in docs]
    except:
        contexts = ["Context retrieval failed"]
    
    return response, contexts