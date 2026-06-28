import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
data = {
    "text": [
        "win money now",
        "hello how are you",
        "free prize claim now",
        "let's meet tomorrow",
        "click here to win",
        "are you coming today"
    ],
    "label": [1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam
}

df = pd.DataFrame(data)
df
X = df["text"]
y = df["label"]

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
msg = ["free money offer now"]
msg_vec = vectorizer.transform(msg)

print(model.predict(msg_vec))
