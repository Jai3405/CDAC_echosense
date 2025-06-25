import streamlit as st
import os
from pathlib import Path
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
import tempfile

# Load .env and configure Gemini
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("gemini-1.5-flash")

# Load models
whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")
embedding_model = SentenceTransformer("all-mpnet-base-v2")

# Transcription function
def transcribe(audio_file):
    segments, _ = whisper_model.transcribe(audio_file.name, language="hi")
    return " ".join(segment.text for segment in segments)

# Translate and correct function using Gemini
@st.cache_data(show_spinner=False)
def translate(text):
    prompt = f"""
    You're a Hindi-English translator. The following text is a Hindi speech transcription.
    Step 1: Correct spelling/grammar/punctuation errors in Hindi.
    Step 2: Translate the corrected Hindi to professional English.

    Hindi Input:
    {text}

    Respond as:
    Hindi (Corrected):
    <corrected_hindi>

    English Translation:
    <translated_english>
    """
    response = model_gemini.generate_content(prompt)
    translated = response.text.split("English Translation:")[-1].strip()
    return translated

# Cosine similarity function
def cosine_sim(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))

# Novelty check via Gemini
@st.cache_data(show_spinner=False)
def check_novelty(comment, previous):
    prompt = f"""
    You are filtering out duplicate opinions.

    Previous accepted unique comments:
    {previous}

    New comment:
    {comment}

    Question: Does this new comment add any novel point or perspective compared to the earlier ones?

    Reply with either:
    - "Duplicate" if it repeats existing ideas
    - "Novel" if it adds a new point
    """
    result = model_gemini.generate_content(prompt).text.strip()
    return "Novel" if "novel" in result.lower() else "Duplicate"

# Streamlit UI
st.set_page_config(page_title="EchoSense", layout="wide")
st.title("🎙️ EchoSense: Intelligent Comment Filtering System")
st.caption("Semantic Relevance + Novelty Detection of Audio Comments")

with st.expander(" Upload Inputs"):
    root_audio = st.file_uploader(" Root Episode (Hindi .mp3)", type="mp3", key="root")
    comment_files = st.file_uploader(" Comment Audio Files (Hindi .mp3)", type="mp3", accept_multiple_files=True, key="comments")
    threshold = st.slider(" Cosine Similarity Threshold for Relevance", 0.0, 1.0, 0.75, step=0.01)

if st.button("Run EchoSense"):
    if not root_audio or not comment_files:
        st.warning("Please upload both Root and Comment audio files.")
    else:
        with st.spinner(" Transcribing + Translating Root Episode..."):
            root_text = transcribe(root_audio)
            root_translated = translate(root_text)
            root_embedding = embedding_model.encode(root_translated)

        accepted = []
        results = []

        st.subheader(" Comment Evaluation")
        for file in comment_files:
            with st.spinner(f" Processing: {file.name}"):
                try:
                    text = transcribe(file)
                    translated = translate(text)
                    emb = embedding_model.encode(translated)
                    sim = cosine_sim(root_embedding, emb)
                    relevant = sim >= threshold

                    if relevant:
                        accepted.append(translated)

                    results.append({
                        "comment": translated,
                        "similarity": sim,
                        "relevant": relevant
                    })

                    st.markdown(f"**Comment:** {translated}\n\n**Similarity:** {sim:.2f} → {' Relevant' if relevant else ' Irrelevant'}")

                except Exception as e:
                    st.error(f"Failed to process {file.name}: {e}")

        if accepted:
            st.subheader(" Novelty Detection (LLM)")
            novel = []
            for i, comment in enumerate(accepted):
                if i == 0:
                    novel.append(comment)
                    st.success(f"Novel: {comment}")
                    continue

                context = "\n".join(novel)
                verdict = check_novelty(comment, context)

                if verdict == "Novel":
                    novel.append(comment)
                    st.success(f"Novel: {comment}")
                else:
                    st.info(f"Duplicate: {comment}")

            st.success(f"Final Accepted Comments: {len(novel)}")
        else:
            st.warning("No relevant comments passed the threshold.")
