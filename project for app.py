
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def preprocess_image(image_path):
  """Loads and preprocesses an image for the Inception model."""
  # Assuming your target image size is 224x224 (common for InceptionV3)
  target_size = (224, 224)

  # Load the image
  img = load_img(image_path, target_size=target_size)

  # Convert to a NumPy array
  img_array = img_to_array(img)

  # Rescale values to be between 0 and 1 (common for InceptionV3)
  img_array = img_array / 255.0

  # Expand dimensions for the model (batch size of 1)
  return np.expand_dims(img_array, axis=0)



from tensorflow.keras.models import load_model

# Load the pre-trained model with frozen weights
inception = load_model('model_inception.h5', compile=False)

# Preprocess your new image (replace with your preprocessing logic)
new_image_path = "deasease.jpg"
new_image = preprocess_image(new_image_path)

# Make prediction
predictions = inception.predict(new_image)
#print(predictions)
# Access the most likely class
predicted_class = np.argmax(predictions[0])

print(f"Predicted class: {predicted_class}")

# Define a dictionary to map predicted class to labels
label_map = {
    0: "Diseased leaf",
    1: "Diseased leaf",
    2: "Fresh leaf",
    3: "Fresh leaf"
}

# Check the predicted class and print corresponding label
if predicted_class in label_map:
    print(label_map[predicted_class])
else:
    print("Unknown class")