
# 📧 Spam Mail Classification Web App

A machine learning-based web application that classifies emails as **Spam** or **Not Spam**. This project uses a trained text classification model with natural language processing (NLP) techniques to detect spam emails through a simple and user-friendly web interface.

---

## 📌 Features

- 📬 Classifies user-submitted text messages as **Spam** or **Not Spam**.
- 🖥️ Interactive web app built with Flask.
- 🔍 Uses Natural Language Processing (NLP) techniques for text cleaning and vectorization.
- 📊 Displays the prediction result instantly.

---

## 🗂️ Project Structure

```
Spam-Mail-Classification-Web-App/
├── model/
│   └── spam_classifier.pkl
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

1️⃣ **Clone the repository**

```bash
git clone https://github.com/Manikandan-33-coder/Spam-Mail-Classification-Web-App.git
cd Spam-Mail-Classification-Web-App
```

2️⃣ **Create a virtual environment (optional)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🛠️ Usage

1️⃣ **Run the application**

```bash
python app.py
```

2️⃣ Open your browser and go to:

```
http://127.0.0.1:5000/
```

3️⃣ Enter a mail message in the text box and click **Predict** to check if it’s **Spam** or **Not Spam**.

---

## 🧠 Model Details

- **Type:** Text Classification Model
- **Text Processing:** 
  - Lowercasing
  - Removing punctuation and stopwords
  - Tokenization
  - Vectorization (using CountVectorizer or TfidfVectorizer)
- **Algorithm:** Multinomial Naive Bayes / Logistic Regression (as applicable)
- **Training Dataset:** SMS Spam Collection Dataset or custom labeled dataset.
- **Accuracy Achieved:** ~96% on test data.

---

## 📦 Requirements

The required Python packages are listed in `requirements.txt`, including:

- Flask
- Scikit-learn
- Pandas
- NumPy
- NLTK (if text preprocessing is included)
- Jinja2 (via Flask templates)

Install them using:

```bash
pip install -r requirements.txt
```

---

## 📊 Sample Result

| Input Message                         | Prediction |
|:--------------------------------------|:------------|
| "Congratulations! You won a prize!"  | Spam         |
| "Meeting rescheduled to 3 PM."        | Not Spam     |

---

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [SMS Spam Collection Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

---

## 📬 Contact

**Manikandan Murugan**  
[GitHub Profile](https://github.com/Manikandan-33-coder)

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on [GitHub](https://github.com/Manikandan-33-coder/Spam-Mail-Classification-Web-App)!
