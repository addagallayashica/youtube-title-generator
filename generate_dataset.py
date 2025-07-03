import pandas as pd
import random
from faker import Faker
import os

fake = Faker()

# Templates
title_templates = [
    "Top Tips for {}",
    "How to Master {}",
    "Complete Guide to {}",
    "Things You Must Know About {}",
    "Why {} is Easier Than You Think",
    "The Ultimate Tutorial on {}"
]

description_templates = [
    "Discover the best strategies for {}, including real-world examples and expert techniques.",
    "Learn everything about {} in this comprehensive guide packed with insights.",
    "This video breaks down {}, making it simple and actionable for beginners and pros.",
    "Find out how to achieve success in {} with these proven methods and tools.",
    "A detailed look into {}, helping you build skills and avoid common pitfalls."
]

# Generate rows
def generate_row():
    prompt = fake.sentence(nb_words=6).replace(".", "")
    title = random.choice(title_templates).format(prompt)
    description = random.choice(description_templates).format(prompt)
    return {"prompt": prompt, "title": title, "description": description}

# Create dataset
data = [generate_row() for _ in range(4000)]
df = pd.DataFrame(data)

# Split
train_df = df.sample(frac=0.9, random_state=42)
test_df = df.drop(train_df.index)

# Save
os.makedirs("data", exist_ok=True)
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("✅ Regenerated train.csv and test.csv with 4000 creative rows.")
