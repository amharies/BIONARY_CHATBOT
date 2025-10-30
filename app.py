# app.py

import streamlit as st
import os
from dotenv import load_dotenv
from query_pipeline import handle_user_query  # <- comes from your Colab backend

# Load environment variables (Gemini + Neon)
load_dotenv()

# --- Page setup ---
st.set_page_config(page_title="Club Knowledge Search Agent", page_icon="💡")
st.title("💡 Club Knowledge Search Agent")
st.write("Ask me anything about your club's events! Type below:")

# --- Chat session setup ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).markdown(msg["content"])

# --- User input ---
if prompt := st.chat_input("Ask your question here..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.spinner("Thinking..."):
        try:
            response = handle_user_query(prompt)
        except Exception as e:
            response = f"⚠️ Error: {e}"

    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.chat_message("assistant").markdown(response)
