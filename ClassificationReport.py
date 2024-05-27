# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import label_binarize
# from sklearn.metrics import roc_curve, auc
# from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt
#
# # Load the data
# data = pd.read_csv("/Users/Shreyan/Downloads/Emotion_detection_with_CNN-main/fer2013.csv")
#
# # Convert pixels to numpy arrays (assuming 'pixels' column and space-separated pixel values)
# data['pixels'] = data['pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' '))
#
# # Assuming 'emotion' is the label column and data needs to be binarized for binary classification
# # For multi-class, you would use label_binarize
# y = label_binarize(data['emotion'].values, classes=[0,1])  # Update classes for your case
# n_classes = y.shape[1]
#
# # Prepare the feature data
# X = np.vstack(data['pixels'].values)
# X = X.reshape(-1, 48 * 48)  # Adjust based on your actual image size
#
# # Split into train and test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load the data from the CSV file
data = pd.read_csv("/Users/Shreyan/Downloads/Emotion_detection_with_CNN-main/fer2013.csv")

# Filter training data
training_data = data[data["Usage"] == "Training"]

# Separate features (pixels) and target variable (emotion)
X_train = training_data["pixels"].apply(lambda x: np.fromstring(x, sep=" "))  # Convert pixel string to array
y_train = training_data["emotion"]

# Split training data further (optional, can be done during model training)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train a model (replace this with your actual classification model)
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# model.fit(X_train, y_train)

# Assuming you already have a trained model (replace with your prediction logic)
y_pred = model.predict(X_val)  # Replace X_val with testing data if split further

# Generate the classification report
print("Classification Report:\n")
print(classification_report(y_val, y_pred))

# Generate the confusion matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_val, y_pred))
