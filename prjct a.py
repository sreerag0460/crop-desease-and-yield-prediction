from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob


import matplotlib.pyplot as plt


# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'train'
valid_path = 'val'





# Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG

import ssl
import urllib.request

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context
# Here we will be using imagenet weights

inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)



# don't train existing weights
for layer in inception.layers:
    layer.trainable = False


# useful for getting number of output classes
folders = glob('train/*')

# our layers - you can add more if you want
x = Flatten()(inception.output)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=inception.input, outputs=prediction)


# view the structure of the model
model.summary()

len(folders)

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('train',target_size = (224, 224),batch_size = 32,class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('test',target_size = (224, 224),batch_size = 32,class_mode = 'categorical')

# fit the model
# Run the cell. It will take some time to execute
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=20,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

# save it as a h5 file


from tensorflow.keras.models import load_model

model.save('model_inception.h5')

y_pred = model.predict(test_set)

y_pred

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