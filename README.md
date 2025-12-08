# Agentic-AI-for-Insights
Welcome to our AI Studio Project with Break Through Tech and 7-11!

# **7-11A TEAM**

| Name                                    | GitHub Handle         | Contribution                                                                                     |
| --------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------ |
| Derek Le                                | anhle1203             | Chunk Text + Cross-Document Trend Analysis + Human Evaluation (Recall@5, Precision@5)           |
| Janice Kennedy                          | 0921-janice           | Embeddings + FAISS Database + LangGraph/LangChain Workflow                                       |
| Noor Kanaan                             | n00rkanaan            | Chunk Text + Human Evaluation (Recall@5, Precision@5) + README                                   |
| Paromita Talukder                       | BlazedDonuts          | Semantic Search + FAISS Database + LangGraph/LangChain Workflow                                  |
| Sanjanaa Viswanathan                    | Sanjanaa12            | Parse PDFs + BGE Reranker + Cross-Document Trend Analysis                                        |
| Shreya Singaravel                       | shreyasing12          | Parse PDFs + Multi-Document Querying + Human Evaluation (Recall@5, Precision@5)                  |
| Will Li                                 | w-lly                 | Embeddings + FAISS Database + BGE Reranker + Multi-Document Querying + Streamlit App + HuggingFace Deployment |
| AI Studio Coach: Aram Ramos             | aramseries            | Guidance, architecture feedback                                                                  |
| Advisor: Sai Sandeep Kantareddy         | SaiSandeepKantareddy  | Technical support, modeling recommendations                                                      |



## **Project Highlights**

* Built an agentic multi-step Retrieval-Augmented Generation (RAG) system capable of synthesizing insights from complex enterprise PDFs.
* Designed a hybrid retrieval method combining BM25 keyword search and FAISS vector embeddings for improved accuracy.
* Implemented a LangGraph-based workflow (Planner → Retriever → Grader → Synthesizer) for structured reasoning and validated outputs.
* Enhanced retrieval precision using a BGE reranker, improving contextual relevance before synthesis.
* Delivered a web-based demo interface allowing natural-language queries over thousands of PDF-based document chunks.

---

## :) **Setup and Installation**

* - **Clone the repository**
  - `git clone https://github.com/w-lly/Agentic-AI-for-Insights.git`
  - `cd Agentic-AI-for-Insights`

- **Create and activate a virtual environment**
  - macOS / Linux:
    - `python3 -m venv env`
    - `source env/bin/activate`
  - Windows:
    - `python3 -m venv env`
    - `env\Scripts\activate`

- **Install dependencies**
  - `pip install -r .\requirements.txt`
  - **Key libraries used:**
    - PyMuPDF  
    - FAISS  
    - SentenceTransformers  
    - LangChain / LangGraph  
    - Transformers (for reranker)

- **Launch the frontend demo**
  - `cd .\streamlit-app\`
  - `streamlit run app.py` 


--- 

## **Project Overview**

* The project addresses 7-Eleven’s need for instant accessibility to large volumes of internal documentation.
* **The challenge:** manual analysis of large unstructured document collections is slow and inconsistent.
* **The goal:** build a prototype that demonstrates multi-step reasoning and provides clear, accurate insights across diverse document types.

---

## **Data Exploration**

Dataset
  * Documents were parsed using PyMuPDF and chunked using GPT-3.5 Turbo.
  * Chunking improved embedding readability and reduced over-compression of meaning.
  * The dataset consisted of enterprise PDF manuals converted into text chunks.
  * Embeddings generated using all-MiniLM-L6-v2 (384-dimensional vectors).
  * Embeddings stored in a FAISS index for fast similarity search.

## **Model Development**

Retrieval Methods
  * BM25 keyword retriever: matches query terms directly.
  * FAISS embedding retriever: finds conceptually similar text.
  * Ensemble retriever: blends both methods (weights = 0.5, 0.5).

Workflow
  * Planner → Retriever → Grader → Synthesizer (via LangGraph & LangChain)
  * BGE reranker used to optimize retrieval before grading.

Evaluation
  * Used RAGAS Context Precision with GROQ-hosted LLaMA-3.3-70B.
  
---

## **Results & Key Findings**

Test 1
  * Question: How to fill the powder in the coffee machine?
  * Score: 0.125
  * So, lower performance on this type of instruction-based query.

Test 2
  * Question: Error Message: No milk or just milk foam
  * Score: 0.8056
  * So, high accuracy and strong retrieval on troubleshooting queries.

 * What we found was that the model performs well on certain queries but struggles with others depending on context density and retrieval quality.

---

## **Next Steps**
Improving retrieval quality for difficult instructional queries

---

## **Live Demo**
The Space is deployed at [https://huggingface.co/spaces/w-lly/Agentic-AI-for-Insights-App](https://huggingface.co/spaces/w-lly/Agentic-AI-for-Insights-App)

---

## **License**
This project is licensed under the MIT License.

---

## **Acknowledgements**
  * 7-Eleven (host company)
  * Aram Ramos — AI Studio Coach, for his invaluable guidance and mentorship
  * Sai Sandeep Kantareddy — Challenge Advisor, for his support, insights, and encouragement
