# 🩺 MedMCQA RAG Chatbot (Ollama + LangGraph)

A fully local, Retrieval-Augmented Generation (RAG) medical chatbot that uses:

- ✅ MedMCQA medical QA dataset
- ✅ ChromaDB for vector similarity search
- ✅ SentenceTransformers for embeddings
- ✅ Ollama (e.g. `llama3`, `phi3`, `mistral`) as the LLM
- ✅ LangGraph for flexible, modular pipeline orchestration
- ✅ Streamlit UI for simple interaction

---

## 📁 Folder Structure

```
MEDMCQA_chatbot/
├── app.py                # Streamlit UI
├── medmcq_chatbot.py     # Dataset, preprocessing, and vector store
├── rag_graph.py          # LangGraph flow with Ollama
├── requirements.txt
└── README.md
```

---

## 🧰 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/tecblic-shreshang/MEDMCQA_chatbot.git
cd MEDMCQA_chatbot
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and start Ollama

Install Ollama from: [https://ollama.com](https://ollama.com)

Then pull a model and start it:

```bash
ollama pull llama3
ollama run llama3
```

---

## 🚀 Run the App

```bash
streamlit run app.py
```

Then open your browser to [http://localhost:8501](http://localhost:8501)

---

## 💡 How It Works

### ✅ LangGraph Structure

```
User Input → Retriever (ChromaDB)
            ↓
       Context Builder
            ↓
      LLM (Ollama via LangChain)
            ↓
       Final Answer
```

* `rag_graph.py` defines this pipeline using LangGraph’s StateGraph.
* `medmcq_chatbot.py` loads and embeds questions from the MedMCQA dataset.
* `app.py` provides a Streamlit-based frontend.

### ✅ Why Ollama?

* No API keys needed
* Runs offline
* Fast LLM inference
* Easy to switch between models (`llama3`, `phi3`, `mistral`)

---

## 🧪 Example Prompts

* What is the normal blood pressure?
* Which vitamin deficiency causes scurvy?
* What are the symptoms of diabetes?

---

## ⚠️ Disclaimer

This chatbot is for **educational purposes only** and is not intended to provide real medical advice. Always consult a healthcare professional.

---

## 🤝 Contributions

Pull requests and issues are welcome!
