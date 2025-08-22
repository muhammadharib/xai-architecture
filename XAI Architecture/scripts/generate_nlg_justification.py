import json

# Load system description
with open("system_description.txt", "r", encoding="utf-8") as f:
    system_description = f.read()

# Load SHAP explanation
with open("shap_explanation.json", "r", encoding="utf-8") as f:
    shap_data = json.load(f)

predicted_architecture = shap_data.get("predicted_architecture", "Unknown")
top_features = shap_data.get("top_contributing_features", [])

# Format SHAP dimensions into text
formatted_dimensions = ", ".join([f"dimension {feat['dimension']}" for feat in top_features[:5]])

# Generate prompt
prompt = f"""
You are a software architecture expert.

System Description:
{system_description}

Based on this system, the predicted architectural style is: {predicted_architecture}.

The top contributing sentence embedding dimensions were: {formatted_dimensions}.

Now, provide a justification in natural language for why the predicted architecture is suitable for this system. Explain it in simple but technical terms.
"""

# For now, just simulate the response (or call your model here)
justification = f"""
**Solution:**

The predicted architecture for the described system is **{predicted_architecture}**.

This is supported by the contribution of key semantic dimensions from the requirement embeddings, such as {formatted_dimensions}, which reflect patterns in the system’s need for user interaction, data processing, and flexibility.

The system must handle frequent user input, real-time financial updates, and modular insights — making {predicted_architecture} a suitable choice due to its ability to support responsive interfaces and maintainable component boundaries.

This architecture aligns with the system’s functional and non-functional goals.
"""

with open("nlg_justification_phi2.txt", "w", encoding="utf-8") as f:
    f.write(justification.strip())

print("\nJustification generated and saved to `nlg_justification_phi2.txt`")
