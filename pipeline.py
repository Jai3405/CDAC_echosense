import os
import json
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

import google.generativeai as genai
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# 🔐 Get API Key securely
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env or Streamlit secrets.")

# 🔧 Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# 🧠 Load Sentence Embedding Model
embedding_model = SentenceTransformer("all-mpnet-base-v2")

# 🧠 Load Whisper ASR
whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")

# 🔤 Transcription
def transcribe(audio_path):
    segments, _ = whisper_model.transcribe(audio_path, language="hi")
    return " ".join([seg.text for seg in segments])

# 🌐 Gemini Correction + Translation
def correct_and_translate(hindi_text):
    prompt = f"""
You're a Hindi-English translator. The following text is a Hindi speech transcription.
Step 1: Correct spelling/grammar/punctuation errors in Hindi.
Step 2: Translate the corrected Hindi to professional English.

Hindi Input:
{hindi_text}

Respond as:
Hindi (Corrected):
<corrected_hindi>

English Translation:
<translated_english>
"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# 🧮 Embedding Generator
def get_embeddings(text_list):
    return embedding_model.encode(text_list, convert_to_tensor=False)

# 🔍 Cosine Relevance Check
def cosine_similarity_check(root_embeds, comment_embeds, threshold=0.7):
    similarities = cosine_similarity(comment_embeds, root_embeds)
    decisions = []

    for idx, row in enumerate(similarities):
        max_sim = float(np.max(row))
        result = {
            "max_similarity": max_sim,
            "decision": "ACCEPT" if max_sim >= threshold else "REJECT"
        }
        decisions.append(result)

    return decisions

# 🌟 LLM Novelty Detection
def llm_novelty_check(accepted_comments):
    novel_comments = []
    log = []

    for idx, current in enumerate(accepted_comments):
        if idx == 0:
            novel_comments.append(current)
            log.append({
                "comment": current,
                "decision": "NOVEL",
                "reason": "First comment – assumed novel by default"
            })
            continue

        previous_context = "\n".join(novel_comments)

        prompt = f"""
You are a helpful research assistant. We are filtering out duplicate opinions from a discussion.

Below are earlier accepted unique comments:
{previous_context}

Now consider this new comment:
{current}

Question: Does this new comment add any novel point or perspective compared to the earlier ones?

Reply with either:
- "Duplicate" if it repeats existing ideas
- "Novel" if it adds a new point
"""

        try:
            response = gemini_model.generate_content(prompt)
            answer = response.text.strip().lower()

            if "novel" in answer:
                novel_comments.append(current)
                decision = "NOVEL"
            else:
                decision = "DUPLICATE"

            log.append({
                "comment": current,
                "decision": decision,
                "llm_response": response.text.strip()
            })

        except Exception as e:
            log.append({
                "comment": current,
                "decision": "ERROR",
                "error": str(e)
            })

    return novel_comments, log
