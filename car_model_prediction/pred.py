import cv2
import numpy as np
from tkinter import filedialog
from keras.models import model_from_json

# Load the model architecture from the JSON file
with open('model/model.json', 'r') as json_file:
    model_json = json_file.read()

# Create the Keras model from the JSON configuration
model = model_from_json(model_json)

# Load the model weights from the H5 file
model.load_weights('model/model_weights.h5')

# Define class names
class_names = ['AM General Hummer SUV 2000', 'Acura RL Sedan 2012', 'Acura TL Sedan 2012', 'Acura TL Type-S 2008', 'Acura TSX Sedan 2012']

# Function to predict car model from an image
def predict_car_model():
    # Ask the user to select an image file
    filename = filedialog.askopenfilename(initialdir="testImages", title="Select an Image", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])

    if filename:
        # Load and preprocess the image
        image = cv2.imread(filename)
        img = cv2.resize(image, (64, 64))
        img = np.array(img)
        img = img.reshape(1, 64, 64, 3).astype('float32') / 255.0

        # Make a prediction using the loaded model
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class]

        # Display the prediction result
        print(f"Predicted Car Model: {predicted_class_name}")

        # Display the image with the prediction result
        img = cv2.imread(filename)
        img = cv2.resize(img, (800, 400))
        cv2.putText(img, f'Car Model Predicted as: {predicted_class_name}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow('Car Model Prediction', img)
        cv2.waitKey(0)
    else:
        print("No image selected.")

# Call the prediction function
predict_car_model()
