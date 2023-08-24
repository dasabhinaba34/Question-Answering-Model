from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import PyPDF2

app = Flask(__name__)

# Load the QA model and tokenizer
model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Define a maximum segment length
max_segment_length = 400  # Adjust as needed

# Read text from PDF (update the path)
def read_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

pdf_path = "files/Story.pdf"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        question = request.form["question"]

        # Split the text into segments
        pdf_text = read_pdf(pdf_path)
        segments = [pdf_text[i:i+max_segment_length] for i in range(0, len(pdf_text), max_segment_length)]

        # Initialize an empty answer list
        answers = []

        # Process each segment
        for segment in segments:
            inputs = tokenizer(question, segment, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)

            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            answer_span = inputs["input_ids"][0][answer_start:answer_end]
            answer = tokenizer.decode(answer_span, skip_special_tokens=False)
            answers.append(answer)

        # Combine answers from all segments
        combined_answer = " ".join(answers)

        return render_template("index.html", question=question, answer=combined_answer)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
