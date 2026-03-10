# AI-AGENT-FOR-HOSPITAL-amp-MEDICAL-SHOP-PRESCRIPTION-GUIDANCE
# Hospital AI Agent for Medicine Guidance

Production-style hybrid RAG assistant for:
- medicine knowledge Q&A (semantic retrieval over ChromaDB)
- medicine stock checks (structured CSV lookup)

## Architecture
- `Stock path`: user query -> extractor -> fuzzy match -> CSV stock response
- `Knowledge path`: user query -> embeddings -> Chroma retrieval -> LLM answer generation

## Project Structure
- `backend/config.py`: global config
- `backend/utils/csv_loader.py`: CSV repository and fuzzy matching
- `backend/utils/medicine_extractor.py`: intent + medicine extraction (HF model)
- `backend/ingestion/ingest_csv.py`: CSV-to-Chroma ingestion
- `backend/rag/retriever.py`: Chroma retriever
- `backend/rag/llm_model.py`: HuggingFace answer generation model
- `backend/rag/rag_pipeline.py`: retrieval + generation pipeline
- `backend/services/medicine_service.py`: query router and orchestrator
- `frontend/app.py`: Streamlit UI

## Setup
```bash
pip install -r requirements.txt
```

## Ingest Data
```bash
python -m backend.ingestion.ingest_csv --clear
```

## Run Streamlit UI
```bash
streamlit run frontend/app.py
```
