import os
import glob
import json
import openai

# Set your OpenAI API key here or via environment variable OPENAI_API_KEY
openai.api_key = os.environ.get("OPENAI_API_KEY", "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")

def label_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Construct a prompt that instructs GPT-4 to label the file
    prompt = f"""You are a labeling assistant.
Read the following file content and output a JSON dictionary with the labeling result.
{content}
    
Output only the JSON dictionary."""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    result_text = response["choices"][0]["message"]["content"]
    try:
        result_json = json.loads(result_text)
    except Exception as e:
        print(f"Error parsing JSON for file {file_path}: {e}")
        result_json = {}
    return result_json

def main():
    current_dir = os.path.dirname(__file__)
    labeling_dir = os.path.join(current_dir, "labeling")
    output_file = os.path.join(current_dir, "results.json")
    
    results = []
    for file_path in glob.glob(os.path.join(labeling_dir, "*.txt")):
        label = label_file(file_path)
        results.append({"file": os.path.basename(file_path), "label": label})
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Labels saved in {output_file}")

if __name__ == "__main__":
    main()