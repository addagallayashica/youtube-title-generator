# scripts/train.py

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import TextDataset
import torch
import os

def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

def main():
    # Load tokenizer and model
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Add padding token if needed
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    # Load datasets
    train_dataset = load_dataset("data/train.txt", tokenizer)
    eval_dataset = load_dataset("data/test.txt", tokenizer)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir="./model_output",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="epoch",  # renamed from evaluation_strategy
        logging_dir="./logs",
        save_total_limit=2,
        logging_steps=100,
        save_strategy="epoch",
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
        prediction_loss_only=True,
        fp16=torch.cuda.is_available(),  # use float16 if GPU available
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save final model
    model_dir = "model_output"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"✅ Model training complete and saved to {model_dir}/")

if __name__ == "__main__":
    main()
