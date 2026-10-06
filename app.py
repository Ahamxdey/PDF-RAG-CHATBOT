from flask import Flask, render_template, request, jsonify
import os
import time
import re
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai

# =========================
# LOAD ENVIRONMENT
# =========================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env file")
else:
    print("GEMINI API KEY FOUND")

gemini_client = genai.Client(
    api_key=API_KEY
)

# =========================
# FLASK APP
# =========================

app = Flask(__name__, static_folder="static")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================
# PDF STORAGE
# =========================

pdf_chunks = []

# Store last question and answer
# Avoid unnecessary repeated API calls
last_question = ""
last_answer = ""


# =========================
# TEXT CLEANING
# =========================

def clean_text(text):

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================
# SPLIT PDF INTO CHUNKS
# =========================

def split_text(text, chunk_size=1200):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# =========================
# SIMPLE PDF RETRIEVAL
# =========================

def retrieve_relevant_chunks(question, chunks, top_k=5):

    question_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            question.lower()
        )
    )

    scored_chunks = []

    for chunk in chunks:

        chunk_words = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                chunk.lower()
            )
        )

        score = len(
            question_words.intersection(chunk_words)
        )

        scored_chunks.append(
            (score, chunk)
        )

    scored_chunks.sort(
        key=lambda x: x[0],
        reverse=True
    )

    relevant = [
        chunk
        for score, chunk in scored_chunks[:top_k]
        if score > 0
    ]

    return relevant


# =========================
# HOME
# =========================

@app.route("/")
def home():

    return render_template("index.html")


# =========================
# UPLOAD PDF
# =========================

@app.route("/upload", methods=["POST"])
def upload_pdf():

    global pdf_chunks
    global last_question
    global last_answer

    if "pdf" not in request.files:

        return jsonify({
            "message": "No PDF selected."
        })

    file = request.files["pdf"]

    if file.filename == "":

        return jsonify({
            "message": "Please select a PDF."
        })

    if not file.filename.lower().endswith(".pdf"):

        return jsonify({
            "message": "Only PDF files are allowed."
        })

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(file_path)

    try:

        reader = PdfReader(file_path)

        full_text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                full_text += clean_text(
                    page_text
                ) + " "

        if not full_text.strip():

            return jsonify({
                "message": "PDF text could not be extracted."
            })

        pdf_chunks = split_text(
            full_text,
            chunk_size=1200
        )

        # Reset previous question
        last_question = ""
        last_answer = ""

        print("--------------------------------")
        print("PDF UPLOADED SUCCESSFULLY")
        print("FILE:", file.filename)
        print("PAGES:", len(reader.pages))
        print("CHUNKS:", len(pdf_chunks))
        print("--------------------------------")

        return jsonify({
            "message": "PDF uploaded successfully!"
        })

    except Exception as e:

        print("PDF ERROR:", e)

        return jsonify({
            "message": "Error reading PDF."
        })


# =========================
# ASK QUESTION
# =========================

@app.route("/ask", methods=["POST"])
def ask_question():

    global pdf_chunks
    global last_question
    global last_answer

    data = request.get_json()

    if not data:

        return jsonify({
            "answer": "Please enter a question."
        })

    question = data.get(
        "question",
        ""
    ).strip()

    if not question:

        return jsonify({
            "answer": "Please enter a question."
        })

    if not pdf_chunks:

        return jsonify({
            "answer": "Please upload a PDF first."
        })

    # =========================
    # SAME QUESTION CACHE
    # =========================

    if question.lower() == last_question.lower():

        print("CACHE USED - NO NEW GEMINI REQUEST")

        return jsonify({
            "answer": last_answer
        })

    # =========================
    # RETRIEVE PDF CONTENT
    # =========================

    relevant_chunks = retrieve_relevant_chunks(
        question,
        pdf_chunks,
        top_k=5
    )

    if not relevant_chunks:

        answer = (
            "not have in pdf sir"
        )

        last_question = question
        last_answer = answer

        return jsonify({
            "answer": answer
        })

    context = "\n\n".join(
        relevant_chunks
    )

    # =========================
    # PROMPT
    # =========================

    prompt = f"""
You are a PDF-only question answering assistant.

Answer the question ONLY using the PDF content provided below.

IMPORTANT RULES:

1. Use ONLY the provided PDF content.
2. Do NOT use outside knowledge.
3. Do NOT guess or invent information.
4. If the answer is not available in the provided PDF content, reply exactly:

"I couldn't find the answer in the uploaded PDF."

5. Give a clear and simple answer.

========================
RELEVANT PDF CONTENT
========================

{context}

========================
QUESTION
========================

{question}

========================
ANSWER
========================
"""

    # =========================
    # GEMINI REQUEST
    # =========================

    try:

        response = gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        answer = response.text

        last_question = question
        last_answer = answer

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        print("GEMINI ERROR:", e)

        # =========================
        # 503 RETRY ONCE
        # =========================

        if "503" in str(e):

            print("Gemini busy. Waiting 8 seconds...")

            time.sleep(8)

            try:

                response = gemini_client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=prompt
                )

                answer = response.text

                last_question = question
                last_answer = answer

                return jsonify({
                    "answer": answer
                })

            except Exception as retry_error:

                print(
                    "GEMINI RETRY ERROR:",
                    retry_error
                )

                return jsonify({
                    "answer":
                    "Gemini is temporarily busy. Please wait a few seconds and try again."
                })

        return jsonify({
            "answer":
            "Gemini is temporarily unavailable. Please try again."
        })


# =========================
# RUN APPLICATION
# =========================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )