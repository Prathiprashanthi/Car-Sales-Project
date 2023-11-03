import cv2
import numpy as np
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

# Function to predict car model from an image file
def predict_car_model(image_file_path):
    # Load and preprocess the image
    image = cv2.imread(image_file_path)
    img = cv2.resize(image, (64, 64))
    img = np.array(img)
    img = img.reshape(1, 64, 64, 3).astype('float32') / 255.0

    # Make a prediction using the loaded model
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class]

    return predicted_class_name

# Provide the path to the image file you want to predict
image_file_path = 'car model\testImages\a1.jpg'  # Replace with your image file path
predicted_model = predict_car_model(image_file_path)

# Display the predicted car model
print(f"Predicted Car Model: {predicted_model}")
