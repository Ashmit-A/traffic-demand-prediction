from preprocess import load_data
from preprocess import preprocess

from feature_engineering import feature_engineering

df = load_data("data/raw/train.csv")

print("Original Shape:")
print(df.shape)

df = preprocess(df)

df = feature_engineering(df)

print("\nFinal Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nPreview:")
print(df.head())