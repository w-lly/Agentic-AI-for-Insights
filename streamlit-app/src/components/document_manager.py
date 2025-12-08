"""
Document Manager Component
"""

import streamlit as st

from components.document_tabs.upload_tab import render_upload_tab
from components.document_tabs.view_tab import render_view_tab
from components.document_tabs.statistics_tab import render_statistics_tab

def render_document_manager():
    """Render the document management interface"""
    
    st.header("Document Management")
    
    tab1, tab2, tab3 = st.tabs(["Upload Documents", "View Documents", "Document Statistics"])
    
    with tab1:
        render_upload_tab()
    
    with tab2:
        render_view_tab()
    
    with tab3:
        render_statistics_tab()