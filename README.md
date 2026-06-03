# AI Car Price Researcher & MCP Agent

A used car recommendation and valuation system that lets you search naturally and find out if a listing is worth the asking price. Built with FastAPI, LangChain, LangGraph, ChromaDB, and a trained ML model on the backend.

Two ways to use it — a terminal-based MCP agent and a FastAPI web app — both running on the same core engine.

---

## What It Does

### 🔍 Natural Language Car Search

Just type what you're looking for:

> *"Budget SUV in Mumbai"* or *"Automatic diesel under 8 lakhs in Pune"*

The system filters listings, runs a vector similarity search, and surfaces the most relevant matches along with a short explanation of why each car fits your query.

### 💰 Price Valuation

Paste in a car's details and get an instant verdict — underpriced, fairly priced, or overpriced — compared to predicted market value. Powered by a Random Forest Regressor trained on used car listings.

```
Hyundai,Creta,2020,Petrol,Automatic,28000,12.5
```

### 🧠 Conversational Memory (Terminal Agent)

The agent keeps context across messages in a session, so you don't have to repeat yourself:

```
You: budget SUV
You: automatic
Agent: [searches for budget automatic SUVs]
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Backend | Python, FastAPI |
| AI / Agents | LangChain, LangGraph (ReAct), Ollama (Qwen2.5:3B) |
| RAG | ChromaDB, HuggingFace Sentence Transformers |
| ML | Scikit-learn, Random Forest Regressor |
| Frontend | HTML, CSS, Jinja2 |

---

## Project Structure

```
project/
├── agent.py           # MCP terminal agent
├── app.py             # FastAPI web app
├── rag.py             # Vector search logic
├── model.py           # Valuation inference
├── train_model.py     # Model training
├── chat_memory.py     # Session memory
├── listings.csv
├── car_price_model.pkl
├── encoders.pkl
├── chroma_db/
└── templates/
    └── index.html
```

---

## Getting Started

### Terminal Agent

```bash
ollama run qwen2.5:3b
python agent.py
```

### Web App

```bash
uvicorn app:app --reload
```

Visit `http://127.0.0.1:8000`

---

## Sample Queries

**Search**
```
budget suv mumbai
automatic diesel pune
electric suv
family car under 8 lakhs
```

**Price Check**
```
Hyundai,Creta,2020,Petrol,Automatic,28000,12.5
```

*End-to-end AI engineering project — RAG, agentic workflows, ML regression, and a full-stack web interface, all in one.*
