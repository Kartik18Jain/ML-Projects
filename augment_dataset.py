import pandas as pd

# Load original dataset
data = pd.read_excel(
    "dataset/AI-Powered Chatbot.xlsx"
)

# New training examples for the grades intent
new_examples = [
    # =========================
    # GRADES
    # =========================

    {
        "User Message": "How can I appeal a grade?",
        "Intent": "grades",
        "Bot Response": "Grade appeals should be submitted according to the university's academic appeals procedure."
    },
    {
        "User Message": "I want to dispute my marks.",
        "Intent": "grades",
        "Bot Response": "You can request a review of your grades through the appropriate academic procedure."
    },
    {
        "User Message": "My grade is incorrect. What should I do?",
        "Intent": "grades",
        "Bot Response": "If you believe your grade is incorrect, contact the appropriate instructor or academic department."
    },
    {
        "User Message": "How do I challenge my grade?",
        "Intent": "grades",
        "Bot Response": "Grade challenges should follow the university's grade appeal procedure."
    },
    {
        "User Message": "Can I request a grade review?",
        "Intent": "grades",
        "Bot Response": "You can request a grade review according to the university's academic procedures."
    },
    {
        "User Message": "I want to appeal my exam grade.",
        "Intent": "grades",
        "Bot Response": "You can follow the university's procedure for appealing an exam grade."
    },
    {
        "User Message": "How can I get my grade reconsidered?",
        "Intent": "grades",
        "Bot Response": "Grade reconsideration should be requested through the appropriate academic procedure."
    },
    {
        "User Message": "What is the process for appealing marks?",
        "Intent": "grades",
        "Bot Response": "Please follow the university's academic grade appeal procedure."
    },
    {
        "User Message": "Can I contest my final grade?",
        "Intent": "grades",
        "Bot Response": "You can contest a final grade through the university's applicable academic appeal process."
    },

    # =========================
    # EXAM SCHEDULE
    # =========================

    {
        "User Message": "When are my exams?",
        "Intent": "exam_schedule",
        "Bot Response": "Please check the official university examination schedule for your exam dates."
    },
    {
        "User Message": "What is the exam schedule?",
        "Intent": "exam_schedule",
        "Bot Response": "Please check the official university examination schedule for the latest exam dates."
    },
    {
        "User Message": "When is my final exam?",
        "Intent": "exam_schedule",
        "Bot Response": "Please refer to the official examination schedule for your final exam date."
    },
    {
        "User Message": "When are the midterm exams?",
        "Intent": "exam_schedule",
        "Bot Response": "Please check the official examination schedule for the midterm exam dates."
    },
    {
        "User Message": "Where can I find the exam dates?",
        "Intent": "exam_schedule",
        "Bot Response": "The examination dates are available in the official university examination schedule."
    },
    {
        "User Message": "When will my exams take place?",
        "Intent": "exam_schedule",
        "Bot Response": "Please check the official examination schedule for your exam dates."
    },
    {
        "User Message": "I want to know my exam dates.",
        "Intent": "exam_schedule",
        "Bot Response": "Please check the official university examination schedule for your exam dates."
    },
    {
        "User Message": "Tell me about the upcoming exams.",
        "Intent": "exam_schedule",
        "Bot Response": "Please refer to the official examination schedule for upcoming exam dates."
    }
]

# Convert new examples to DataFrame
new_data = pd.DataFrame(new_examples)

# Keep only columns that exist in the original dataset
new_data = new_data[
    ["User Message", "Intent", "Bot Response"]
]

# Add missing columns
for column in data.columns:
    if column not in new_data.columns:
        new_data[column] = ""

# Arrange columns like original dataset
new_data = new_data[data.columns]

# Combine original + new examples
training_data = pd.concat(
    [data, new_data],
    ignore_index=True
)

# Save new dataset
training_data.to_excel(
    "dataset/training_data.xlsx",
    index=False
)

print("================================")
print("Training dataset created!")
print("================================")

print("Original samples:", len(data))
print("Added samples:", len(new_data))
print("Total samples:", len(training_data))

print("\nNew grades examples:")
print(
    training_data[
        training_data["Intent"] == "grades"
    ][["User Message", "Intent"]].to_string(index=False)
)