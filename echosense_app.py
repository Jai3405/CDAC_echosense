import streamlit as st
import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Load sentence transformer model
embedder = SentenceTransformer("all-mpnet-base-v2")

st.set_page_config(page_title="EchoSense", layout="centered")
st.title("🎙️ EchoSense: Intelligent Comment Filtering")

st.write("Upload a Root Episode transcript and a list of comment transcripts to analyze relevance and novelty.")

# Upload root episode and comments
root_text = st.text_area("Root Episode (translated):", height=150)
comments_input = st.text_area("Comment Transcripts (translated, one per line):", height=200)

threshold = st.slider("Cosine Similarity Threshold (Relevance)", 0.0, 1.0, 0.7, 0.01)

if st.button("Analyze Comments"):
    if not root_text or not comments_input:
        st.warning("Please input both root episode and comment transcripts.")
    else:
        root_embedding = embedder.encode([root_text])[0]
        comments = [c.strip() for c in comments_input.splitlines() if c.strip()]
        comment_embeddings = embedder.encode(comments)

        results = []
        accepted_comments = []

        st.subheader(" Relevance Filtering")
        for comment, embedding in zip(comments, comment_embeddings):
            sim_score = util.cos_sim(root_embedding, embedding).item()
            relevant = sim_score >= threshold
            results.append({
                "comment": comment,
                "similarity": sim_score,
                "relevant": relevant
            })
            if relevant:
                accepted_comments.append(comment)
            st.markdown(f"**Comment:** {comment}\n\n*Similarity:* {sim_score:.2f} → {' Relevant' if relevant else ' Irrelevant'}")

        st.subheader(" Novelty Detection (LLM)")
        novel_comments = []
        for idx, current in enumerate(accepted_comments):
            if idx == 0:
                novel_comments.append(current)
                st.success(f" Novel: {current}")
                continue
            previous_context = "\n".join(novel_comments)
            prompt = f"""
            You are filtering out duplicate opinions.
            Previous accepted unique comments:
            {previous_context}

            New comment:
            {current}

            Is this new comment novel (adds new point)? Reply with only 'Novel' or 'Duplicate'.
            """
            try:
                response = gemini_model.generate_content(prompt).text.strip().lower()
                if "novel" in response:
                    novel_comments.append(current)
                    st.success(f" Novel: {current}")
                else:
                    st.info(f" Duplicate: {current}")
            except Exception as e:
                st.error(f"⚠️ Error processing comment: {current} - {e}")

        st.subheader(" Summary")
        st.markdown(f"**Total Comments:** {len(comments)}")
        st.markdown(f"**Accepted (Relevant):** {len(accepted_comments)}")
        st.markdown(f"**Novel Comments:** {len(novel_comments)}")
