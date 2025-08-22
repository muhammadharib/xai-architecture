import json
import shap
import numpy as np
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

# ---------- Load Data ----------
def load_labeled_dataset(path="labeled_case_studies.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_target_case(path="ai_generated_requirements_clean.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- Convert Requirements to Text ----------
def flatten_requirements(case):
    fr_text = " ".join([req["title"] + " " + req["description"] for req in case["functional_requirements"]])
    nfr_text = " ".join([req["title"] + " " + req["description"] for req in case["non_functional_requirements"]])
    return fr_text + " " + nfr_text

# ---------- Main Pipeline ----------
def main():
    print("\nRunning shap_architecture_explainer.py ...\n")
    labeled_cases = load_labeled_dataset()
    target_case = load_target_case()

    # Flatten requirements into text
    X_texts = [flatten_requirements(case) for case in labeled_cases]
    y_labels = [case["architecture_label"] for case in labeled_cases]
    target_text = flatten_requirements(target_case)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels)

    # Sentence embeddings
    print("[*] Generating sentence embeddings ...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X_embed = model.encode(X_texts)
    X_target_embed = model.encode([target_text])

    # Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_embed, y_encoded)

    # Predict
    pred_index = clf.predict(X_target_embed)[0]
    pred_label = label_encoder.inverse_transform([pred_index])[0]
    print(f"Predicted Architecture: {pred_label}")

    # SHAP Explanation
    explainer = shap.Explainer(clf, X_embed)
    shap_values = explainer(X_target_embed)

    # Get SHAP values for predicted class (LogReg gives class-wise SHAP)
    if shap_values.values.ndim == 3:  # multiclass case
        shap_for_pred = shap_values.values[0, pred_index]
    else:  # binary case
        shap_for_pred = shap_values.values[0]

    # Top 10 contributing dimensions
    top_dims = sorted(
        enumerate(shap_for_pred),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]

    explanation = {
        "predicted_architecture": pred_label,
        "top_contributing_features": [
            {"dimension": int(idx), "shap_value": float(score)}
            for idx, score in top_dims
        ]
    }

    with open("shap_explanation.json", "w", encoding="utf-8") as f:
        json.dump(explanation, f, indent=2)

    print("\nTop SHAP Dimensions (harder to interpret, but more accurate):")
    for dim in explanation["top_contributing_features"]:
        print(f"  Dim {dim['dimension']} (SHAP = {dim['shap_value']:.5f})")

    print(" SHAP explanation saved to shap_explanation.json")

if __name__ == "__main__":
    main()
