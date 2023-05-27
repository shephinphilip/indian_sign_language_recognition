import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyaudio
import pytsx3

# Load the saved model
model = load_model('path/to/model.h5')

# Define the classes for the output labels
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Define the function to preprocess the input image
def preprocess_image(image):
    # Resize the image to 224x224
    image = cv2.resize(image, (224, 224))
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normalize pixel values to the range [0, 1]
    image = image / 255.0
    # Add a batch dimension to the image
    image = np.expand_dims(image, axis=0)
    # Add a channel dimension to the image
    image = np.expand_dims(image, axis=3)
    # Return the preprocessed image
    return image

# Define the function to recognize hand gestures from camera
def recognize_gesture():
    # Open the default camera
    cap = cv2.VideoCapture(0)
    # Initialize PyTTSX3 engine
    tts_engine = pytsx3.init()

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Preprocess the input image
        image = preprocess_image(frame)

        # Make a prediction using the model
        prediction = model.predict(image)

        # Get the predicted class label
        label = classes[np.argmax(prediction)]

        # Display the predicted class label on the screen
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the camera frame on the screen
        cv2.imshow('Camera', frame)

        # Speak the recognized text using PyTTSX3
        tts_engine.say(label)
        tts_engine.runAndWait()

        # If the 'q' key is pressed, exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to recognize hand gestures from camera and speech the recognized text
recognize_gesture()
