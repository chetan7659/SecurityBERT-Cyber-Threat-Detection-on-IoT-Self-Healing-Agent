import json

file_path = r'c:\Users\kaush\Desktop\LLM Threat Detection on IIOT\SecurityBERT CLAUDE\notebooks\25_final_evaluation.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if "ACCENT='#7c6cfa'; GREEN='#4ecca3'; RED='#fc5c65'; YELLOW='#f7b731'" in line:
                line = line.replace("YELLOW='#f7b731'", "YELLOW='#f7b731'; ORANGE='#fd9644'")
            new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Successfully updated ORANGE definition in notebook.")
