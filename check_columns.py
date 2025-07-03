
import pandas as pd

df = pd.read_csv("data/train.csv")
print("📋 Columns in train.csv:")
print(df.columns.tolist())
