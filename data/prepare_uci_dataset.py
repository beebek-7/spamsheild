import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(BASE_DIR, "data")

raw_path = os.path.join(data_dir, "SMSSpamCollection")
out_path = os.path.join(data_dir, "sms_spam.csv")

if not os.path.exists(raw_path):
    raise FileNotFoundError(f"Could not find {raw_path}. Make sure SMSSpamCollection is in the data/ folder.")

# The original file is tab-separated with no header: label \t text
df = pd.read_csv(raw_path, sep="\t", header=None, names=["label", "text"], encoding="utf-8")

# Basic cleanup: drop empty texts
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].reset_index(drop=True)

print("Dataset overview:")
print(df["label"].value_counts())
print(f"\nTotal messages: {len(df)}")

# Save in the format used by your project: label,text
df.to_csv(out_path, index=False)
print(f"\nSaved cleaned dataset to: {out_path}")
