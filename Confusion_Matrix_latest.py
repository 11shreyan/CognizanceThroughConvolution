# # Gives the confusion matrix of all seven emotions individually
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Load the dataset
# file_path = 'emotions.csv'
# emotions_df = pd.read_csv(file_path)
#
# # Extract features and labels
# emotions_df['Timestamp'] = pd.to_datetime(emotions_df['Timestamp'])
# emotions_df['Timestamp'] = emotions_df['Timestamp'].map(pd.Timestamp.timestamp)
#
# # Additional feature engineering (example: time-based features)
# emotions_df['hour'] = pd.to_datetime(emotions_df['Timestamp'], unit='s').dt.hour
# emotions_df['day'] = pd.to_datetime(emotions_df['Timestamp'], unit='s').dt.day
# emotions_df['month'] = pd.to_datetime(emotions_df['Timestamp'], unit='s').dt.month
# X = emotions_df[['Timestamp', 'hour', 'day', 'month']]
# y = emotions_df['Emotion']
#
# # Encode the labels
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
#
# # Standardize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # Define a function to generate classification report and confusion matrix for each emotion
# def generate_classification_reports_and_confusion_matrices(X, y, label_encoder):
#     reports = {}
#     confusion_matrices = {}
#     for emotion in label_encoder.classes_:
#         # Create binary labels for one-vs-all classification
#         y_binary = (y == label_encoder.transform([emotion])[0]).astype(int)
#
#         # Split the data into training and test sets
#         X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)
#
#         # Train a more sophisticated classifier
#         classifier = RandomForestClassifier(n_estimators=100, random_state=42)
#         classifier.fit(X_train, y_train)
#
#         # Generate predictions
#         y_pred = classifier.predict(X_test)
#
#         # Generate the classification report
#         report = classification_report(y_test, y_pred, target_names=['Not ' + emotion, emotion], zero_division=1)
#         reports[emotion] = report
#
#         # Generate the confusion matrix
#         cm = confusion_matrix(y_test, y_pred)
#         confusion_matrices[emotion] = cm
#
#     return reports, confusion_matrices
#
# # Generate the classification reports and confusion matrices
# classification_reports, confusion_matrices = generate_classification_reports_and_confusion_matrices(X_scaled, y_encoded, label_encoder)
#
# # Print the reports and plot the confusion matrices
# for emotion, report in classification_reports.items():
#     print(f"Classification report for emotion: {emotion}")
#     print(report)
#     print("--------------------------------------------------\n")
#
# for emotion, cm in confusion_matrices.items():
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Not ' + emotion, emotion], yticklabels=['Not ' + emotion, emotion])
#     plt.title(f"Confusion Matrix for emotion: {emotion}")
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Train a more sophisticated classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Generate predictions
y_pred = classifier.predict(X_test)

# Generate the classification report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix for All Emotions")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()