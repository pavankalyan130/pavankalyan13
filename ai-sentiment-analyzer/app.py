"""AI sentiment analyzer using TF-IDF and logistic regression."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

texts = [
    "The product quality is excellent and delivery was fast",
    "I love the user friendly interface",
    "Customer support solved my issue quickly",
    "Amazing experience, I will buy again",
    "The app is simple and very helpful",
    "This is the worst purchase I have made",
    "Delivery was late and the package was damaged",
    "I am disappointed with the customer service",
    "The application crashes every time I open it",
    "Poor quality and not worth the price",
    "Great value for money and easy to use",
    "The team was helpful and professional",
    "I want a refund because this did not work",
    "The instructions were confusing and incomplete",
    "Very satisfied with the performance",
    "The product stopped working after one day",
]
labels = [
    "positive", "positive", "positive", "positive", "positive",
    "negative", "negative", "negative", "negative", "negative",
    "positive", "positive", "negative", "negative", "positive", "negative",
]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

model = Pipeline([
    ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Validation accuracy: {accuracy:.0%}")

examples = [
    "The dashboard is excellent and saved me a lot of time",
    "The service was slow and the results were inaccurate",
    "I am happy with the support team",
]

for message in examples:
    sentiment = model.predict([message])[0]
    confidence = max(model.predict_proba([message])[0])
    print(f"{sentiment.title():8} ({confidence:.0%}) — {message}")
