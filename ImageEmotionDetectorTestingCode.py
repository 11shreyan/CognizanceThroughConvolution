import cv2
import numpy as np
from keras.models import model_from_json

# Dictionary mapping numeric labels to emotion names
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load JSON and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# Path to the input image
image_path = '/Users/Shreyan/Downloads/Emotion_detection_with_CNN-main/TestForImages/chris_pratt_surprise.png'

# Read the input image
frame = cv2.imread(image_path)
frame = cv2.resize(frame, (600, 600))

# Convert the image to grayscale
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Load the Haar Cascade for face detection
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Detect faces in the image
num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

# Process each face found in the image
for (x, y, w, h) in num_faces:
    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
    roi_gray_frame = gray_frame[y:y + h, x:x + w]
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

    # Predict the emotions
    emotion_prediction = emotion_model.predict(cropped_img)
    maxindex = int(np.argmax(emotion_prediction))
    cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

# Display the output image
cv2.imshow('Emotion Detection', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
