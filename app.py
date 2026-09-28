import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai

load_dotenv()

app = Flask(__name__, template_folder="templates")

MODEL_NAME = "gemini-3.1-flash-lite"
CONFIG_PATH = Path(__file__).with_name("chatbot_config.txt")

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY is not set in the environment.")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def load_system_prompt():
    return CONFIG_PATH.read_text(encoding="utf-8")


SYSTEM_PROMPT = load_system_prompt()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    prompt = f"""{SYSTEM_PROMPT}

USER MESSAGE:
{message}

Answer according to the instructions above."""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        return jsonify({"reply": response.text})
    except Exception:
        return jsonify({
            "error": "Sorry, I couldn't process your request right now."
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
