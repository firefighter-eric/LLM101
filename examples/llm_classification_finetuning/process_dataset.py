import pandas as pd
from loguru import logger
import json

df = pd.read_csv("data/llm-classification-finetuning/train.csv")
print(df.head())
dataset = df.to_dict(orient="records")
logger.info(f"Number of records: {len(dataset)}")
with open("data/llm-classification-finetuning/train.json", "w") as f:
    json.dump(dataset, f, indent=4)