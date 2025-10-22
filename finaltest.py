import pandas as pd
from pathlib import Path # To manipulate file paths easier as objects rather than just strings
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from preprocess import preprocess_text
from feature_extraction import extract_features

csv_path = Path(__file__).resolve().parent / "fake-reviews.csv"

df = pd.read_csv(csv_path)
df["cleaned_text"] = df["text_"].apply(preprocess_text)
include_pos = True

df = extract_features(df, include_pos)
output_path = Path(__file__).resolve().parent / "processed-dataset.csv"

df.to_csv(output_path, index = False)
print(f"Processed dataset saved to {output_path}")