import pandas as pd

# Load dataset
data = pd.read_excel("dataset/AI-Powered Chatbot.xlsx")

print("===== DATASET OVERVIEW =====")
print(f"Rows: {data.shape[0]}")
print(f"Columns: {data.shape[1]}")

# Missing values
print("\n===== MISSING VALUES =====")
print(data.isnull().sum())

# Unique intents
print("\n===== UNIQUE INTENTS =====")
print("Number of intents:", data["Intent"].nunique())
print(data["Intent"].unique())

# Intent distribution
print("\n===== INTENT DISTRIBUTION =====")
print(data["Intent"].value_counts())

# Sentiment distribution
print("\n===== SENTIMENT DISTRIBUTION =====")
print(data["Sentiment Label"].value_counts())

# Duplicate user messages
print("\n===== DUPLICATE USER MESSAGES =====")
print(
    "Duplicate messages:",
    data["User Message"].duplicated().sum()
)

# Message length
data["Message Length"] = data["User Message"].str.len()

print("\n===== MESSAGE LENGTH =====")
print(data["Message Length"].describe())