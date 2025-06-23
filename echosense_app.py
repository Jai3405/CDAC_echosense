import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import os
import json
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# Load environment variables (Gemini API Key)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Load SentenceTransformer
embedder = SentenceTransformer("all-mpnet-base-v2")

st.set_page_config(page_title="EchoSense: Intelligent Comment Filtering", layout="wide")
st.title("🎙️ EchoSense: Intelligent Comment Filtering System")
st.caption("Semantic Relevance + Novelty Detection of Audio Comments")

# Input: Root and Comment
with st.expander("📌 Upload Inputs"):
    root_text = st.text_area("🧠 Root Episode (Translated)", height=200)
    comment_input = st.text_area("💬 Comment Input (Translated)", height=200)
    trigger = st.button("🚀 Run EchoSense")

# Outputs after processing
if trigger:
    if not root_text.strip() or not comment_input.strip():
        st.error("❗ Please provide both Root Episode and Comment input.")
    else:
        # Embedding and cosine similarity
        root_embedding = embedder.encode([root_text], normalize_embeddings=True)
        comment_embedding = embedder.encode([comment_input], normalize_embeddings=True)
        similarity_score = float(util.cos_sim(comment_embedding, root_embedding)[0][0])

        SIM_THRESHOLD = 0.72  # Tuned threshold
        relevance = "ACCEPTED ✅" if similarity_score >= SIM_THRESHOLD else "REJECTED ❌"

        st.subheader("🔍 Relevance Check (Cosine Similarity)")
        st.write(f"**Similarity Score:** {similarity_score:.3f}")
        st.write(f"**Decision:** {relevance}")

        if relevance.startswith("ACCEPTED"):
            with st.spinner("🔄 Performing LLM-based Novelty Check..."):
                # Gemini LLM Novelty Prompt
                novelty_prompt = f"""
You are a helpful assistant filtering out repetitive ideas in a discussion. 
Given this root episode:

{root_text}

Now consider the following user comment:

{comment_input}

Is this comment introducing a *novel* point or is it *repeating* ideas already expressed in the root?

Respond with:
- "Novel" if it's a new point
- "Duplicate" if it's repeated.
"""
                try:
                    response = gemini_model.generate_content(novelty_prompt)
                    verdict = response.text.strip()
                    st.subheader("🧠 LLM Novelty Check")
                    st.write(f"**Verdict:** {verdict}")
                except Exception as e:
                    st.error(f"LLM error: {e}")
        else:
            st.info("LLM check skipped as comment is irrelevant to the discussion.")
