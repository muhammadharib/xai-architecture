# 🧠 XAI-Driven Software Architecture Prediction Pipeline

This project provides a fully automated and explainable pipeline for predicting the most suitable software architecture (e.g., MVC, Event-Driven, Microservices, Layered) for a system, based solely on its high-level description.

---

## 🚀 Overview

**Given a natural language system description, this tool:**
1. 🧾 Generates structured software requirements (functional & non-functional).
2. 🧠 Predicts the best-fit architecture using a trained Logistic Regression model.
3. 🪄 Explains the prediction using SHAP (SHapley Additive exPlanations).
4. 📝 Generates a natural language justification using a language model (Phi-2).

---

## 📁 Directory Structure

├── scripts/
│ ├── run_xai_architecture_pipeline.py # Master script (entry point)
│ ├── requirement_generator.py # Generates requirements using Phi-2
│ ├── predict_and_save_architecture.py # ML model prediction
│ ├── shap_architecture_explainer.py # SHAP explanation of prediction
│ ├── generate_nlg_justification.py # NLG for justification
│
├── ai_generated_requirements_clean.json # Auto-generated requirements
├── predicted_architecture.json # Predicted architecture class
├── shap_explanation.json # SHAP values and features
├── nlg_justification_phi2.txt # Final textual explanation
├── system_description.txt # User input


💡 Models Used
Component	                    |   Model
Requirements & Justification	|   Phi-2
                                |
Sentence Embeddings	            |   all-MiniLM-L6-v2 via sentence-transformers
Architecture Predictor	        |   Logistic Regression
Explainability	                |   SHAP Explainer


▶️ Running the Pipeline

From the scripts/ folder:
python run_xai_architecture_pipeline.py
