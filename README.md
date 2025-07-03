# 🎥 YouTube Title & Description Generator

This project is an **AI-based text generation tool** that generates YouTube video titles and descriptions based on a user's input prompt (video topic). It's built using **Hugging Face Transformers** and fine-tuned on a custom dataset of 2,000 rows.

> 🚀 Prompt in → 📌 Title + 📝 Description out

---

## ✨ Features

- 🧠 Fine-tuned GPT-2 model on a custom YouTube dataset
- 📝 Input a video topic and get a creative title + description
- 📂 Clean file structure for training, data prep, and inference
- ✅ Works entirely offline after training

---

## 📁 Project Structure

youtube-gen_model/
├── data/
│ ├── train.csv
│ ├── test.csv
│ ├── train.txt
│ └── test.txt
├── model_output/ # Saved model checkpoint
├── scripts/
│ ├── prepare_data.py # Format CSV to training text
│ ├── train.py # Fine-tune GPT-2
│ ├── generate.py # Inference: generate title + description
│ └── generate_dataset.py # (Optional) Script to create dataset
├── .gitignore
├── README.md
└── requirements.txt
