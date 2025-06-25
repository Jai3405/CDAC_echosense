# echosense_app.py
import streamlit as st
import tempfile
import os
from pathlib import Path
from pipeline import (
    transcribe_audio,
    correct_and_translate,
    get_embeddings,
    cosine_similarity_matrix,
    check_novelty_with_llm
)

# UI Layout
st.set_page_config(page_title="EchoSense - Intelligent Comment Filter", layout="wide")
st.title("🎙️ EchoSense: Intelligent Comment Filtering for Audio Discussions")

st.markdown("""
Upload one root audio episode and multiple comment audio responses.
The system will transcribe, translate, filter relevant comments based on root topic,
and further eliminate duplicate perspectives using LLM-powered novelty check.
""")

# Upload Inputs
st.subheader("📁 Upload Root Episode (MP3)")
root_file = st.file_uploader("Upload root audio...", type=["mp3"], key="root")

st.subheader("🗣️ Upload Comment Responses (MP3)")
comment_files = st.file_uploader("Upload one or more comment audios...", type=["mp3"], accept_multiple_files=True, key="comments")

st.sidebar.header("⚙️ Settings")
similarity_threshold = st.sidebar.slider("Relevance Threshold (Cosine)", 0.0, 1.0, 0.65, 0.01)

if st.button("🚀 Run EchoSense Pipeline") and root_file and comment_files:
    with st.spinner("🔁 Running ASR + Translation + Relevance + Novelty..."):
        # Temp save files
        tmp_dir = tempfile.TemporaryDirectory()
        root_path = Path(tmp_dir.name) / "root.mp3"
        with open(root_path, "wb") as f:
            f.write(root_file.read())

        comment_paths = []
        for file in comment_files:
            path = Path(tmp_dir.name) / file.name
            with open(path, "wb") as f:
                f.write(file.read())
            comment_paths.append(path)

        # Root processing
        root_transcription = transcribe_audio(root_path)
        root_translation = correct_and_translate(root_transcription)
        root_embedding = get_embeddings([root_translation])

        st.success("✅ Root audio processed")
        st.text_area("📌 Root Episode (Translated)", root_translation, height=100)

        # Comment processing
        accepted_comments = []
        rejected_comments = []
        translated_comments = []

        for path in comment_paths:
            comment_text = transcribe_audio(path)
            translated = correct_and_translate(comment_text)
            translated_comments.append(translated)

        comment_embeddings = get_embeddings(translated_comments)

        # Relevance check
        from numpy import array
        sims = cosine_similarity_matrix(array(comment_embeddings), array(root_embedding))
        for i, sim in enumerate(sims):
            if max(sim) >= similarity_threshold:
                accepted_comments.append(translated_comments[i])
            else:
                rejected_comments.append(translated_comments[i])

        st.info(f"✅ {len(accepted_comments)} comments accepted | ❌ {len(rejected_comments)} rejected")

        # Novelty Check via LLM
        novel = []
        seen = []

        for idx, comment in enumerate(accepted_comments):
            if idx == 0:
                novel.append(comment)
                seen.append(comment)
                continue

            verdict = check_novelty_with_llm(seen, comment)
            if "novel" in verdict.lower():
                novel.append(comment)
                seen.append(comment)

        st.success(f"🧠 Novelty Filtering Done. {len(novel)} novel comments retained.")

        st.subheader("📝 Final Output")
        for i, c in enumerate(novel):
            st.markdown(f"**{i+1}.** {c}")

    tmp_dir.cleanup()

elif st.button("🚀 Run EchoSense Pipeline"):
    st.warning("⚠️ Please upload both root and at least one comment file to proceed.")
