from flask import Flask, render_template, request
import fitz  # PyMuPDF
from transformers import pipeline

app = Flask(__name__)

# Load AI Summarizer Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files["newspaper_pdf"]
        if uploaded_file.filename != "":
            file_path = "uploaded.pdf"
            uploaded_file.save(file_path)

            # Extract text from PDF
            raw_text = extract_text_from_pdf(file_path)

            # Summarize text (limit to 1024 tokens for best results)
            summarized_text = summarizer(raw_text[:1024], max_length=200, min_length=50, do_sample=False)[0]['summary_text']

            return render_template("index.html", summary=summarized_text)

    return render_template("index.html", summary=None)

if __name__ == "__main__":
    app.run(debug=True)


