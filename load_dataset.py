import pandas as pd

# Load the dataset
data = pd.read_excel("dataset/AI-Powered Chatbot.xlsx")

# Display the first 5 rows
print("First 5 rows:")
print(data.head())

# Display column names
print("\nColumn names:")
print(data.columns)

# Display number of rows and columns
print("\nDataset shape:")
print(data.shape)

# Display information about the dataset
print("\nDataset information:")
print(data.info())