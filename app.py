from flask import Flask, render_template, request
from gensim.models import Word2Vec
import joblib
import numpy as np

app = Flask(__name__)

# Load models
w2v_model = Word2Vec.load("spam_word2vec.model")
model = joblib.load("spam_classifier_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Function to get average vector for a message
def get_average_vector(sentence, model):
    words = sentence.lower().split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        message = request.form["message"]
        vector = get_average_vector(message, w2v_model).reshape(1, -1)
        prediction = model.predict(vector)[0]
        result = label_encoder.inverse_transform([prediction])[0]
        return render_template("index.html", prediction=result, message=message)

if __name__ == "__main__":
    app.run(debug=True)
