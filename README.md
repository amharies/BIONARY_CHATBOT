# Club Knowledge Search Agent

A Streamlit-based chatbot that answers questions about Bionary Club events using a custom RAG (Retrieval-Augmented Generation) system.

### Technologies
- Google Gemini API
- Neon (PostgreSQL with pgvector)
- Sentence Transformers (BGE Model)
- LangChain
- Streamlit

### To Deploy on Streamlit Cloud
1. Push these files to GitHub.
2. Go to https://share.streamlit.io → New App → select this repo.
3. Add Secrets:
   - GEMINI_API_KEY
   - NEON_DB_URL
4. Click Deploy.
