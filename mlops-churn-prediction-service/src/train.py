from pathlib import Path
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.features import make_training_data

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

X, y = make_training_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)

metrics = {
    "accuracy": round(float(accuracy_score(y_test, predictions)), 3),
    "precision": round(float(precision_score(y_test, predictions)), 3),
    "recall": round(float(recall_score(y_test, predictions)), 3),
}
joblib.dump(model, ARTIFACTS / "churn_model.joblib")
(ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("Training complete:", metrics)
