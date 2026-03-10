"""
Flask REST API for audio quality comparison.
"""

import os
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify, render_template

from audio_processor import aggregate_quality_metrics

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "input" not in request.files or "reference" not in request.files:
        return jsonify({"error": "Both 'input' and 'reference' audio files are required."}), 400

    input_file = request.files["input"]
    ref_file = request.files["reference"]

    if not _allowed_file(input_file.filename):
        return jsonify({"error": f"Unsupported input file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400
    if not _allowed_file(ref_file.filename):
        return jsonify({"error": f"Unsupported reference file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    input_suffix = Path(input_file.filename).suffix or ".wav"
    ref_suffix = Path(ref_file.filename).suffix or ".wav"

    input_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix)
    ref_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ref_suffix)

    try:
        input_file.save(input_tmp.name)
        ref_file.save(ref_tmp.name)
        input_tmp.close()
        ref_tmp.close()

        results = aggregate_quality_metrics(input_tmp.name, ref_tmp.name)
        return jsonify(results)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        for path in (input_tmp.name, ref_tmp.name):
            try:
                os.unlink(path)
            except OSError:
                pass


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug)
