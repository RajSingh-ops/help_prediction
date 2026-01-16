import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("data.csv")

texts = df["text"]
labels = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Vectorize text
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=15000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train classifier
model = LogisticRegression(max_iter=2000, class_weight="balanced")
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Predict on new text
def predict(text):
    v = vectorizer.transform([text])
    return model.predict(v)[0]

print(predict("do no one loves me whats the point of my worth less life idk man"))
