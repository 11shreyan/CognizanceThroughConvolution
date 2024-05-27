# Emotion_detection_with_CNN


### Packages need to be installed
- keras==2.13.1
- matplotlib==3.8.4
- numpy==1.24.3
- numpy==1.24.4
- opencv_python==4.9.0.80
- pandas==2.0.3
- scikit_learn==1.4.2
- seaborn==0.13.2
- tensorflow==2.13.0
- tensorflow_macos==2.13.0

### download FER2013 dataset
- from below link and put in data folder under your project directory
- https://www.kaggle.com/msambare/fer2013

### Train Emotion detector
- with all face expression images in the FER2013 Dataset
- command --> python TranEmotionDetector_updated.py

It will take several hours depends on your processor. (On M2 processor with 8 GB RAM it took me around 2 hours for 120 epochs)
after Training , you will find the trained model structure and weights are stored in your project directory.
- emotion_model.json
- emotion_model.h5

copy these two files create model folder in your project directory and paste it.

### run your emotion detection test file
- python TestEmotionDetector.py