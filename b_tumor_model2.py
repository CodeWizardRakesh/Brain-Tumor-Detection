import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

# Define the data directories
train_dir = 'D:\\sem3\\final projects\\Brain tumor prediction\\dataset\\Training'
test_dir = 'D:\\sem3\\final projects\\Brain tumor prediction\\dataset\\Testing'

# Set hyperparameters
batch_size = 32
epochs = 10
input_shape = (150, 150, 3)

# Load and preprocess training data
train_images = []
train_labels = []

for class_name in os.listdir(train_dir):
    class_dir = os.path.join(train_dir, class_name)
    class_label = os.listdir(train_dir).index(class_name)

    for image_file in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_file)
        image = load_img(image_path, target_size=input_shape[:2])
        image = img_to_array(image)
        image /= 255.0  # Normalize to [0, 1]

        train_images.append(image)
        train_labels.append(class_label)

train_images = np.array(train_images)
train_labels = to_categorical(train_labels, num_classes=len(os.listdir(train_dir)))

# Load and preprocess testing data
test_images = []
test_labels = []

for class_name in os.listdir(test_dir):
    class_dir = os.path.join(test_dir, class_name)
    class_label = os.listdir(test_dir).index(class_name)

    for image_file in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_file)
        image = load_img(image_path, target_size=input_shape[:2])
        image = img_to_array(image)
        image /= 255.0  # Normalize to [0, 1]

        test_images.append(image)
        test_labels.append(class_label)

test_images = np.array(test_images)
test_labels = to_categorical(test_labels, num_classes=len(os.listdir(test_dir)))

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(os.listdir(train_dir)), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_images, train_labels,
    epochs=epochs,
    validation_data=(test_images, test_labels),
    batch_size=batch_size,
    verbose=1
)

# Save the trained model
model.save('brain_tumor_model2.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)

