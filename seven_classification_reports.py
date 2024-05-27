# 7 different emotions classification report code - Works but it has undefined metrics warning
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import LabelEncoder
#
# # Load the dataset
# file_path = 'emotions.csv'
# emotions_df = pd.read_csv(file_path)
#
# # Extract features and labels
# # For simplicity, we'll use the timestamp as a feature
# emotions_df['Timestamp'] = pd.to_datetime(emotions_df['Timestamp'])
# emotions_df['Timestamp'] = emotions_df['Timestamp'].map(pd.Timestamp.timestamp)
# X = emotions_df[['Timestamp']]
# y = emotions_df['Emotion']
#
# # Encode the labels
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
#
#
# # Define a function to generate classification report for each emotion
# def generate_classification_reports(X, y, label_encoder):
#     reports = {}
#     for emotion in label_encoder.classes_:
#         # Create binary labels for one-vs-all classification
#         y_binary = (y == label_encoder.transform([emotion])[0]).astype(int)
#
#         # Split the data into training and test sets
#         X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)
#
#         # Train a simple classifier
#         classifier = DecisionTreeClassifier(random_state=42)
#         classifier.fit(X_train, y_train)
#
#         # Generate predictions
#         y_pred = classifier.predict(X_test)
#
#         # Generate the classification report
#         report = classification_report(y_test, y_pred, target_names=['Not ' + emotion, emotion])
#         reports[emotion] = report
#     return reports
#
#
# # Generate the classification reports
# classification_reports = generate_classification_reports(X, y_encoded, label_encoder)
#
# # Print the reports
# for emotion, report in classification_reports.items():
#     print(f"Classification report for emotion: {emotion}")
#     print(report)
#     print("--------------------------------------------------\n")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'emotions.csv'
emotions_df = pd.read_csv(file_path)

# Extract features and labels
emotions_df['Timestamp'] = pd.to_datetime(emotions_df['Timestamp'])
emotions_df['Timestamp'] = emotions_df['Timestamp'].map(pd.Timestamp.timestamp)

# Additional feature engineering (example: time-based features)
emotions_df['hour'] = pd.to_datetime(emotions_df['Timestamp'], unit='s').dt.hour
emotions_df['day'] = pd.to_datetime(emotions_df['Timestamp'], unit='s').dt.day
emotions_df['month'] = pd.to_datetime(emotions_df['Timestamp'], unit='s').dt.month
X = emotions_df[['Timestamp', 'hour', 'day', 'month']]
y = emotions_df['Emotion']

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Define a function to generate classification report for each emotion
def generate_classification_reports(X, y, label_encoder):
    reports = {}
    for emotion in label_encoder.classes_:
        # Create binary labels for one-vs-all classification
        y_binary = (y == label_encoder.transform([emotion])[0]).astype(int)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

        # Train a more sophisticated classifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)

        # Generate predictions
        y_pred = classifier.predict(X_test)

        # Generate the classification report
        report_for_classification = classification_report(y_test, y_pred, target_names=['Not ' + emotion, emotion], zero_division=1)
        reports[emotion] = report_for_classification
    return reports


# Generate the classification reports
classification_reports = generate_classification_reports(X_scaled, y_encoded, label_encoder)

# Print the reports
for emotion, report in classification_reports.items():
    print(f"Classification report for emotion: {emotion}")
    print(report)
    print("--------------------------------------------------\n")
