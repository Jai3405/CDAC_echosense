import os
import json
from dotenv import load_dotenv

import google.generativeai as genai
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 🔐 Load secrets
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Models
embedding_model = SentenceTransformer("all-mpnet-base-v2")
whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")

# 🧠 Transcription
def transcribe_audio(audio_path):
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

# 🔍 Embedding
def get_embeddings(texts):
    return embedding_model.encode(texts, convert_to_tensor=False)

# 🔍 Cosine similarity
def cosine_similarity_matrix(comment_embeddings, root_embeddings):
    return cosine_similarity(comment_embeddings, root_embeddings)

# 🧠 LLM Novelty check
def check_novelty_with_llm(previous_comments, new_comment):
    prompt = f"""
You are a helpful research assistant. We are filtering out duplicate opinions from a discussion.

Below are earlier accepted unique comments:
{chr(10).join(previous_comments)}

Now consider this new comment:
{new_comment}

Question: Does this new comment add any novel point or perspective compared to the earlier ones?

Reply with either:
- "Duplicate" if it repeats existing ideas
- "Novel" if it adds a new point
"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()
