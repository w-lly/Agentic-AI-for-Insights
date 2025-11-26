"""
Chat Interface Component
"""

import streamlit as st
from rag.pipeline_manager import query_rag_system

def render_chat_interface():
    """Render the main chat interface"""
    
    st.header("Chat with your Coffee Machine Manual")
    
    # Check if system is initialized
    if not st.session_state.system_initialized:
        st.warning("⚠️ Please initialize the RAG system from the sidebar first.")
        
        # Show example queries
        st.subheader("Example Queries")
        examples = [
            "How do I fill powder in the coffee machine?",
            "What should I do if there's no milk or just milk foam?",
            "How do I clean the machine?",
            "What are the water supply requirements?",
            "How do I empty the grounds container?"
        ]
        
        for example in examples:
            if st.button(f"💡 {example}", key=example):
                st.info("Please initialize the system first to use this query.")
        
        return
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show retrieved contexts for assistant messages
                if message["role"] == "assistant" and "contexts" in message:
                    with st.expander("📚 View Retrieved Context"):
                        for j, context in enumerate(message["contexts"], 1):
                            st.markdown(f"**Context {j}:**")
                            st.text(context[:300] + "..." if len(context) > 300 else context)
                            st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your coffee machine...", 
                               disabled=not st.session_state.system_initialized):
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response, contexts = query_rag_system(prompt)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "contexts": contexts
                    })
                    
                    # Show contexts in expander
                    with st.expander("📚 View Retrieved Context"):
                        for i, context in enumerate(contexts, 1):
                            st.markdown(f"**Context {i}:**")
                            st.text(context[:300] + "..." if len(context) > 300 else context)
                            st.divider()
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Suggested follow-up questions
    if len(st.session_state.messages) > 0:
        st.divider()
        st.subheader("💡 Suggested Follow-ups")
        
        suggestions = [
            "Can you explain that in more detail?",
            "What are the safety precautions?",
            "Are there any maintenance requirements?",
            "What should I do if this doesn't work?"
        ]
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"suggest_{i}"):
                    # Simulate clicking with the suggestion
                    st.session_state.messages.append({"role": "user", "content": suggestion})
                    st.rerun()