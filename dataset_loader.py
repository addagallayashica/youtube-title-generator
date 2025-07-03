import pandas as pd
from datasets import Dataset

def load_dataset(csv_path):
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')  # or use 'latin1' as alias
    df = df.dropna(subset=["title", "description"])
    df["text"] = "Title: " + df["title"] + "\nDescription: " + df["description"]
    return Dataset.from_pandas(df[["text"]])
