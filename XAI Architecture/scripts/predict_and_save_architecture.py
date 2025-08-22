import json

# Scoring rules based on keyword matches
architecture_rules = {
    "Microservices": ["integration", "scalability", "modularity", "services", "independent"],
    "Event-Driven": ["real-time", "event", "updates", "notifications"],
    "MVC": ["interface", "controller", "user interaction", "forms", "navigation"],
    "Layered": ["persistence", "database", "business logic", "data processing", "separation"]
}

def load_requirements(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def score_architectures(requirements):
    scores = {arch: 0 for arch in architecture_rules}

    # Combine FR and NFR descriptions
    all_texts = []
    for req in requirements.get("functional_requirements", []) + requirements.get("non_functional_requirements", []):
        all_texts.append(req["title"].lower())
        all_texts.append(req["description"].lower())

    # Score based on keyword occurrence
    for arch, keywords in architecture_rules.items():
        for keyword in keywords:
            scores[arch] += sum(1 for text in all_texts if keyword in text)

    return scores

def predict_architecture(scores):
    return max(scores, key=scores.get)

def main():
    input_path = "ai_generated_requirements_clean.json"
    data = load_requirements(input_path)

    print(" Loaded requirements.")
    scores = score_architectures(data)
    predicted = predict_architecture(scores)

    print("\n Architecture Scores:")
    for arch, score in scores.items():
        print(f"  {arch}: {score}")

    print(f"\n Predicted Architecture: **{predicted}**")

    # Save result
    result = {
        "predicted_architecture": predicted,
        "architecture_scores": scores
    }
    with open("predicted_architecture.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(" Saved to predicted_architecture.json")

if __name__ == "__main__":
    main()
