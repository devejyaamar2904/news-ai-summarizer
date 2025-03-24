import os
from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "News AI Summarizer is running!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render assigns a dynamic port
    app.run(host="0.0.0.0", port=port)
