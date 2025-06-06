from flask import Flask, request, render_template
from models import model
import json
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
llm = model.init_model()

# Chargement des données de contexte "Warhammer 40k"
with open("/home/a798673/TestAI/static/WarhammerLore.json", "r", encoding="utf-8") as f:
    data = json.load(f)
# Découpage du lore Warhammer 40k en chunks
chunks = [f"{entry['text']}" for entry in data]

# Embedding du contexte pour le modèle
embedder = SentenceTransformer("all-MiniLM-L6-v2")
vectors = embedder.encode(chunks)

@app.route("/")
def home():
    return render_template("home.html", active="home")

# Premier test avec un llm simple
@app.route("/test01", methods=["GET", "POST"])
def test01():
    response = None
    if request.method == "POST":
        prompt = request.form["prompt"]
        response = model.ask(llm, prompt)
    return render_template("test01.html", response=response, active="test01")

# Second test en rajoutant du contexte "Warhammer 40k"
@app.route("/test02", methods=["GET", "POST"])
def test02():
    response = None
    if request.method == "POST":
        prompt = request.form["prompt"]
        response = model.ask(llm, prompt)
    return render_template("test02.html", response=response, active="test02")


@app.route("/about")
def about():
    return render_template("about.html", active="about")

if __name__ == "__main__":
    app.run(debug=True)