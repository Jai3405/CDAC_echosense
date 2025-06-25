# pipeline.py
import os
from pathlib import Path
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate key
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Load models once
whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")
embed_model = SentenceTransformer("all-mpnet-base-v2")

# ASR using FastWhisper
def transcribe_audio(audio_path):
    segments, _ = whisper_model.transcribe(str(audio_path), language="hi")
    return " ".join(segment.text for segment in segments)

# Translate using Gemini 1.5 Flash
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

# Embed using ALL-MPNet-Base
def get_embeddings(texts):
    return embed_model.encode(texts, convert_to_tensor=False)

# Cosine similarity check
def cosine_similarity_matrix(A, B):
    import numpy as np
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return np.dot(A_norm, B_norm.T)

# LLM-based novelty check
def check_novelty_with_llm(previous_comments, current):
    context = "\n".join(previous_comments)
    prompt = f"""
You are a helpful research assistant. We are filtering out duplicate opinions from a discussion.

Below are earlier accepted unique comments:
{context}

Now consider this new comment:
{current}

Question: Does this new comment add any novel point or perspective compared to the earlier ones?

Reply with either:
- "Duplicate" if it repeats existing ideas
- "Novel" if it adds a new point
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"ERROR: {e}"
