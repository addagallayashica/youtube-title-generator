# scripts/generate.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model and tokenizer
model_path = "model_output"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_title_description(prompt):
    input_text = f"Prompt: {prompt}\nTitle:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    output = model.generate(
        input_ids,
        max_length=200,
        temperature=0.9,
        top_p=0.95,
        top_k=40,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        num_return_sequences=1
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Force structure
    title = "❌ Not found"
    description = "❌ Not found"
    if "Title:" in decoded and "Description:" in decoded:
        try:
            title = decoded.split("Title:")[1].split("Description:")[0].strip()
            description = decoded.split("Description:")[1].split("###")[0].strip()
        except:
            pass

    return title, description

# Interactive loop
while True:
    prompt = input("\n📥 Enter video topic (or 'exit'): ").strip()
    if prompt.lower() == 'exit':
        break
    title, description = generate_title_description(prompt)
    print(f"\n📌 Title: {title}\n📝 Description: {description}")
