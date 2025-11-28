# Agentic-AI-for-Insights

Welcome to our AI Studio Project!

This project aims to create an intelligent assistant that can extract and summarize insights from large, unstructured enterprise documents, reducing manual analysis, and improving decision making, by using techniques like Retrieval-Augmented Generation (RAG), vector search and agent orchestration. The goal is to create a prototype that demonstrates multi-step reasoning and evaluates the quality of answers across different document types. Key goals include developing a robust ingestion and retrieval pipeline using a vector database, implementing agent coordination for multi-step cross-document reasoning, and designing an insight generation interface evaluated on retrieval precision (>80%) and human-rated clarity (>4/5). The project will progress through three milestones. First, we will build the foundation by parsing PDFs, processing scanned files with OCR, breaking documents into manageable sections, creating searchable text representations, and storing them in a vector database for fast semantic search. Next, we will add a reasoning layer that allows the system to handle multi-step questions, pull together evidence from multiple documents, and present well-supported answers. Finally, we will focus on insights and usability by adding summarization, trend analysis, and a simple interface, while evaluating the system’s speed, accuracy, and reliability. The project will conclude with a live demo showcasing real queries, results, and recommendations for scaling the approach.

To run streamlit app:

ensure requirements are imported: `pip install -r .\requirements.txt`
cd into `streamlit-app`
run `streamlit run .\app.py`

## Live Demo

The Space is deployed at [https://huggingface.co/spaces/w-lly/Agentic-AI-for-Insights-App](https://huggingface.co/spaces/w-lly/Agentic-AI-for-Insights-App)
