import json

with open("generated_prompts.json") as file:
    print(f"{json.load(file)['00000_00.jpg']}")