from flask import Flask, render_template, request
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from Script_generator import Script

load_dotenv()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate_script", methods=["POST"])
def generate_script():

    topic = request.form["topic"]
    content_type = request.form["type"]
    emotion = request.form["emotion"]
    # Script Generator
    script_response=Script(content_type,topic,emotion)

    return render_template("index.html", script=script_response)

if __name__ == "__main__":
    app.run(debug=True)