import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re

def extract_clean_json(generated_text):
    """
    Extract the first valid top-level JSON object from generated text.
    Fixes trailing commas if needed.
    """
    try:
        if "Answer:" in generated_text:
            generated_text = generated_text.split("Answer:")[1]

        # Track braces to isolate the JSON block
        brace_count = 0
        json_start = -1
        for i, char in enumerate(generated_text):
            if char == '{':
                if brace_count == 0:
                    json_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and json_start != -1:
                    json_str = generated_text[json_start:i + 1]

                    # Remove trailing commas before ] or }
                    json_str = re.sub(r",\s*([\]}])", r"\1", json_str)

                    return json.loads(json_str)

        raise ValueError("Could not find complete JSON block.")

    except json.JSONDecodeError as e:
        print("\n[!] JSON decoding failed. Here's the output that failed:\n")
        print(generated_text)
        raise e


def load_system_description():
    with open("system_description.txt", "r", encoding="utf-8") as f:
        return f.read().strip()


def build_prompt(system_description):
    return f"""
You are a software analyst. Based on the following system description, generate well-structured functional and non-functional requirements in JSON format.

System Description:
{system_description}

Output JSON format:
{{
  "system_description": "...",
  "functional_requirements": [
    {{ "title": "...", "description": "..." }},
    ...
  ],
  "non_functional_requirements": [
    {{ "title": "...", "description": "..." }},
    ...
  ]
}}

Make sure both functional and non-functional requirements are included.

Answer:
"""


def main():
    print("[*] Loading PHI-2 model...")
    model_id = "microsoft/phi-2"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Load input prompt
    system_description = load_system_description()
    prompt = build_prompt(system_description)

    print("[*] Generating Requirements from Model...")

    outputs = generator(prompt, max_new_tokens=600, do_sample=False)
    output = outputs[0]["generated_text"]

    try:
        structured_output = extract_clean_json(output)
    except Exception as e:
        print(" Could not extract valid JSON from model output.")
        raise e

    with open("ai_generated_requirements_clean.json", "w", encoding="utf-8") as f:
        json.dump(structured_output, f, indent=2)
    
    print("Generated output was:\n", output[:500])

    print(" Cleaned and structured requirements saved to ai_generated_requirements_clean.json")


if __name__ == "__main__":
    main()
