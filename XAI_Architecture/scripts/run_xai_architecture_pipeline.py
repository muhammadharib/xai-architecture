import shap
import subprocess
import json
import sys

def ask_description():
    print("Enter your system description:")
    return input("> ").strip()

def save_system_description(system_description):
    with open("system_description.txt", "w", encoding="utf-8") as f:
        f.write(system_description)
    print("System description saved to system_description.txt")

def run_script(script_name):
    print(f"\nRunning {script_name} ...")
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True
        )
        print(f"{script_name} completed.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e)


def display_outputs():
    print("\nFinal Outputs:")

    # Architecture + SHAP
    try:
        with open("shap_explanation.json", "r", encoding="utf-8") as f:
            shap_data = json.load(f)
            print("\nPredicted Architecture:", shap_data.get("predicted_architecture", "N/A"))
            print("Top Contributing Features:")
            for feat in shap_data.get("top_contributing_features", [])[:5]:
                if "dimension" in feat:
                    print(f" - Dimension {feat['dimension']} (SHAP = {feat['shap_value']:.4f})")
                else:
                    print(f" - {feat.get('keyword', 'N/A')} (SHAP = {feat['shap_value']:.4f})")
    except Exception as e:
        print("Could not read shap_explanation.json:", e)

    # NLG Justification
    try:
        with open("nlg_justification_phi2.txt", "r", encoding="utf-8") as f:
            print("\nNLG Justification:\n")
            print(f.read().strip())
    except Exception as e:
        print("Could not read NLG justification:", e)

def main():
    print("XAI Software Architecture Pipeline Runner")

    # Get input
    system_description = ask_description()
    save_system_description(system_description)

    # Run in order
    run_script("requirement_generator.py")
    run_script("predict_and_save_architecture.py")
    run_script("shap_architecture_explainer.py")
    run_script("generate_nlg_justification.py")

    # Final output
    display_outputs()

if __name__ == "__main__":
    main()
