import os
import tempfile
import traceback

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from audio_processor.loader import load_audio, validate_audio
from audio_processor.quality_metrics import compute_quality_report

ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "aiff", "aif", "m4a"}

app = Flask(__name__, static_folder="static")
app.config["UPLOAD_FOLDER"] = tempfile.gettempdir()
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB


def _allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    input_path = None
    ref_path = None
    try:
        # Validate uploaded files
        if "input_audio" not in request.files:
            return jsonify({"error": "Missing file: input_audio"}), 400
        if "reference_audio" not in request.files:
            return jsonify({"error": "Missing file: reference_audio"}), 400

        input_file = request.files["input_audio"]
        ref_file = request.files["reference_audio"]

        if input_file.filename == "":
            return jsonify({"error": "No file selected for input_audio"}), 400
        if ref_file.filename == "":
            return jsonify({"error": "No file selected for reference_audio"}), 400

        if not _allowed_file(input_file.filename):
            return jsonify({"error": f"Unsupported format for input_audio: {input_file.filename}"}), 400
        if not _allowed_file(ref_file.filename):
            return jsonify({"error": f"Unsupported format for reference_audio: {ref_file.filename}"}), 400

        # Save to temp files
        input_filename = secure_filename(input_file.filename)
        ref_filename = secure_filename(ref_file.filename)

        input_path = os.path.join(app.config["UPLOAD_FOLDER"], f"input_{os.getpid()}_{input_filename}")
        ref_path = os.path.join(app.config["UPLOAD_FOLDER"], f"ref_{os.getpid()}_{ref_filename}")

        input_file.save(input_path)
        ref_file.save(ref_path)

        # Load audio – use a common sample rate (22050 Hz) for comparison
        target_sr = 22050
        input_samples, sr = load_audio(input_path, sr=target_sr)
        ref_samples, _ = load_audio(ref_path, sr=target_sr)

        # Compute quality report
        report = compute_quality_report(input_samples, ref_samples, sr)

        return jsonify(report)

    except ValueError as exc:
        return jsonify({"error": f"Invalid audio: {exc}"}), 422
    except Exception:  # noqa: BLE001
        app.logger.error("Unhandled error during analysis:\n%s", traceback.format_exc())
        return jsonify({"error": "Internal processing error. Check server logs for details."}), 500
    finally:
        for path in (input_path, ref_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode, host="0.0.0.0", port=5000)
