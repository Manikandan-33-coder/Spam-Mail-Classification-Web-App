import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from gensim.models import Word2Vec
import numpy as np
import joblib

# 1️⃣ Load dataset
df = pd.read_excel("Mail_New1.xlsx")
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# 2️⃣ Encode target labels
label_encoder = LabelEncoder()
df['v1'] = label_encoder.fit_transform(df['v1'])  # ham=0, spam=1

# 3️⃣ Define features and labels
X = df['v2'].astype(str)
Y = df['v1']

# 4️⃣ Tokenize sentences into words
X_tokenized = X.apply(lambda x: x.lower().split())

# 5️⃣ Train Word2Vec model
w2v_model = Word2Vec(sentences=X_tokenized, vector_size=100, window=5, min_count=1, workers=4, sg=1)

# 6️⃣ Function to average word vectors for a sentence
def get_average_vector(sentence, model):
    vectors = [model.wv[word] for word in sentence if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# 7️⃣ Convert text to vectors
X_vectors = np.array([get_average_vector(text, w2v_model) for text in X_tokenized])

# 8️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectors, Y, test_size=0.2, random_state=42)

# 9️⃣ Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 🔟 Prediction & Evaluation
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))

# 1️⃣1️⃣ Save Word2Vec model
w2v_model.save("spam_word2vec.model")

# 1️⃣2️⃣ Save trained Logistic Regression model
joblib.dump(model, "spam_classifier_model.pkl")

# 1️⃣3️⃣ Save LabelEncoder (important for mapping back ham/spam)
joblib.dump(label_encoder, "label_encoder.pkl")
