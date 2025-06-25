from flask import Flask, render_template, request, redirect, url_for
import os
from pathlib import Path
import tempfile
from pipeline import (
    transcribe_audio,
    correct_and_translate,
    get_embeddings,
    cosine_similarity_matrix,
    check_novelty_with_llm
)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = {}
    if request.method == 'POST':
        # Save root file
        root_file = request.files.get('root_audio')
        comment_files = request.files.getlist('comment_audios')
        threshold = float(request.form.get('threshold', 0.65))

        if not root_file or not comment_files:
            return render_template('index.html', error="Please upload both root and comment audio files.")

        with tempfile.TemporaryDirectory() as tmpdirname:
            root_path = Path(tmpdirname) / 'root.mp3'
            root_file.save(root_path)

            comment_paths = []
            for cfile in comment_files:
                cpath = Path(tmpdirname) / cfile.filename
                cfile.save(cpath)
                comment_paths.append(cpath)

            # Process root
            root_transcript = transcribe_audio(root_path)
            root_translation = correct_and_translate(root_transcript)
            root_embedding = get_embeddings([root_translation])

            # Process comments
            translated_comments = []
            for cpath in comment_paths:
                ctext = transcribe_audio(cpath)
                translation = correct_and_translate(ctext)
                translated_comments.append(translation)

            comment_embeddings = get_embeddings(translated_comments)

            # Relevance
            from numpy import array
            sims = cosine_similarity_matrix(array(root_embedding), array(comment_embeddings), threshold)
            accepted = [translated_comments[i] for i, sim in enumerate(sims) if sim['decision'] == 'ACCEPT']
            rejected = [translated_comments[i] for i, sim in enumerate(sims) if sim['decision'] != 'ACCEPT']

            # Novelty
            novel_comments, _ = check_novelty_with_llm(accepted)

            results = {
                'root': root_translation,
                'accepted': accepted,
                'rejected': rejected,
                'novel': novel_comments,
                'threshold': threshold
            }

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
