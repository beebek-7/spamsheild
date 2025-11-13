import pandas as pd
from pathlib import Path

# Output path: data/spamshield_batch_example_100.csv
OUT_PATH = Path("data") / "spamshield_batch_example_100.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

messages = [
    "You have won a free prize! Click here now!",
    "Are we still meeting tomorrow?",
    "Urgent! Your account has been locked. Verify now.",
    "Lunch at 1pm?",
    "Congratulations! You've been selected for a reward.",
    "Call me when you're free.",
    "Limited time offer! Claim your voucher.",
    "Hey, did you get my message?",
    "Your package has been delayed. Check status.",
    "FREE entry into contest! Reply WIN.",
]

rows = [{"text": messages[i % len(messages)]} for i in range(100)]

df = pd.DataFrame(rows)
df.to_csv(OUT_PATH, index=False)

print(f"Saved 100 messages to: {OUT_PATH}")
