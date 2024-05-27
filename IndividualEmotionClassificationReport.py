import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'emotions.csv'
emotions_df = pd.read_csv(file_path)

# Extract features and labels
# For simplicity, we'll use the timestamp as a feature
emotions_df['Timestamp'] = pd.to_datetime(emotions_df['Timestamp'])
emotions_df['Timestamp'] = emotions_df['Timestamp'].map(pd.Timestamp.timestamp)
X = emotions_df[['Timestamp']]
y = emotions_df['Emotion']

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Train a simple classifier
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Generate predictions
y_pred = classifier.predict(X_test)

# Generate the classification report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)