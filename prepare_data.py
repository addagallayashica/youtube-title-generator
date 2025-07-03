# scripts/prepare_data.py

import pandas as pd

def format_text(df):
    formatted = []
    for _, row in df.iterrows():
        prompt = str(row['prompt']).strip()
        title = str(row['title']).strip()
        description = str(row['description']).strip()
        text = f"Prompt: {prompt}\nTitle: {title}\nDescription: {description}\n###\n"
        formatted.append(text)
    return formatted

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

with open("data/train.txt", "w", encoding="utf-8") as f:
    f.writelines(format_text(train_df))

with open("data/test.txt", "w", encoding="utf-8") as f:
    f.writelines(format_text(test_df))

print("✅ Formatted train.txt and test.txt successfully.")
