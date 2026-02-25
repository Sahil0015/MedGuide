# 🩺 MedGuide — AI Health Companion

**MedGuide** is an intelligent AI-powered application that interprets blood and lab test reports, generates structured summaries, and enables contextual chat about health results.  
It combines **multi-agent reasoning**, **vector-based retrieval (LanceDB)**, and **LLMs (OpenAI GPT + Cohere Reranker)** to deliver factual, explainable, and user-friendly insights.

---

## 🚀 Features

- 🧠 **AI-driven report analysis:** Automatically extracts and analyzes lab report values.
- 📄 **Page-wise interpretation:** Asynchronous agents process each page independently.
- 📊 **Structured final report:** Merges multi-page insights into one comprehensive summary.
- 💬 **Conversational interface:** Chat with your analyzed reports using hybrid RAG retrieval.
- ⚡ **Hybrid search:** Combines **vector** + **keyword** search (LanceDB + Cohere reranker).
- 🧩 **Local knowledge base:** Builds an offline LanceDB store for efficient querying.
- 🧑‍⚕️ **Safe AI design:** Focused on factual, educational insights — not medical advice.

---

## 🏗️ Project Architecture

```
MedGuide/
│
├── app/
│   ├── streamlit_app.py        # Main Streamlit UI
│   ├── app.py                  # FastAPI REST backend
│   ├── main.py                 # CLI pipeline for local testing
│   ├── data/                   # Stores processed text, LanceDB, and uploads
│
├── agents/
│   ├── analyzer_agent.py       # Interprets lab values page-wise
│   ├── chat_agent.py           # Conversational RAG agent
│   ├── document_extraction_agent.py  # Extracts test names, values, ranges
│   ├── final_report_agent.py   # Combines all pages into a summary report
│
├── utils/
│   ├── pdf_extractor.py        # Extracts text from PDFs using PyMuPDF
│   ├── pdf_to_txt.py           # Converts PDFs into text for preprocessing
│
├── vectordb/
│   ├── create_vector_db.py     # Builds LanceDB-based knowledge base
│
├── data/
│   ├── knowledge_base/         # (Ignored in Git) Private extracted data
│   ├── sample_reports/         # Example lab reports
│   ├── lancedb/                # Vector DB storage
│
├── requirements.txt            # Dependencies
├── .gitignore                  # Ignore rules
└── README.md                   # Documentation
```

---

## 🧠 How It Works

### 🔹 Step 1 — Upload
Upload any **PDF-based lab report** via the Streamlit interface.

### 🔹 Step 2 — Extraction
`document_extraction_agent` parses test names, values, and reference ranges.

### 🔹 Step 3 — Analysis
`analyzer_agent` interprets results and gives concise, factual explanations.

### 🔹 Step 4 — Report Generation
`final_report_agent` merges all insights into a single structured health report.

### 🔹 Step 5 — Knowledge Base Creation
`create_vector_db.py` generates a LanceDB vector store for RAG retrieval.

### 🔹 Step 6 — Chat Interaction
`chat_agent` enables conversations with your report using OpenAI GPT + Cohere.

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Sahil0015/MedGuide.git
cd MedGuide
```

### 2️⃣ Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate # On Mac/Linux
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set up environment variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
```

### 5️⃣ Run the Streamlit app
```bash
streamlit run app/streamlit_app.py
```

---

## 🧬 Technologies Used

| Category | Technologies |
|-----------|---------------|
| **Frontend** | Streamlit |
| **LLMs & Agents** | OpenAI GPT, Agno, LangChain |
| **Retrieval & DB** | LanceDB, RedisVL, Cohere Reranker |
| **PDF Processing** | PyMuPDF, Tantivy |
| **Utilities** | Python-dotenv, TQDM, Pandas |

---

## 🧾 Key Files

| File | Purpose |
|------|----------|
| `app/streamlit_app.py` | Main Streamlit interface |
| `app/app.py` | FastAPI REST backend (upload, process, chat endpoints) |
| `app/main.py` | CLI pipeline runner for local testing |
| `vectordb/create_vector_db.py` | Creates LanceDB vector store |
| `utils/pdf_extractor.py` | PDF text extraction |
| `utils/pdf_to_txt.py` | Converts PDF to text |
| `agents/*.py` | Multi-agent logic for report extraction, analysis, and chat |

---

## ⚙️ .gitignore Overview
Your `.gitignore` excludes:
```
data/knowledge_base/
venv/
.env
.cache/
__pycache__/
```
→ Ensuring sensitive or auto-generated data is never uploaded.

---

## 🧪 Example Workflow

1. Launch the Streamlit app.  
2. Input your API keys and initialize.  
3. Upload a lab report (PDF).  
4. AI agents extract, analyze, and summarize results.  
5. Review the generated final report.  
6. Chat interactively with the app about your results.  

---

## ⚖️ Disclaimer
> MedGuide provides AI-generated educational insights based on lab data.  
> It is **not a medical diagnostic tool** — always consult a certified doctor for medical decisions.

---

## 💡 Future Enhancements

- Expand **multi-language report interpretation**.  
- Support **FHIR / HL7** medical data formats.  
- Implement **persistent chat memory** across sessions and report history.  
- Add **user authentication** for personalized report management.  
- Deploy as a **Docker container** on cloud platforms (AWS, GCP, Azure).

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgements

- **OpenAI** for GPT models  
- **Cohere** for Reranker API  
- **LanceDB** for high-speed vector storage  
- **Streamlit** for a smooth frontend experience  
- **Agno** and **LangChain** for agent orchestration  

---

### 💖 Contributions

Pull requests and suggestions are welcome!  
If you’d like to contribute, please fork the repository and create a PR.

---

### 🔗 Connect

👨‍💻 **Author:** [Sahil Aggarwal](https://www.linkedin.com/in/sahil-codes/)  
📂 **GitHub:** [Sahil0015](https://github.com/Sahil0015)  
✉️ **Email:** sahilaggarwal1532003@gmail.com
