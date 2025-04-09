import os
import numpy as np
import cv2
import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_and_preprocess_images(folder, img_size=28):
    images = []
    labels = []

    for label_folder in sorted(os.listdir(folder)):
        label_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_path) and len(label_folder) == 1 and label_folder.isalpha():
            for file in os.listdir(label_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(label_path, file)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        resized_img = cv2.resize(img, (img_size, img_size))
                        resized_img = resized_img.reshape(img_size, img_size, 1)
                        images.append(resized_img / 255.0)
                        labels.append(ord(label_folder.upper()) - ord('A'))
    images = np.array(images)
    labels = np.array(labels)
    print(f"Loaded {len(images)} images.")
    return images, labels

def show_example(images, labels, index):
    plt.imshow(images[index].squeeze(), cmap='gray')
    plt.title(f"Example Label: {chr(labels[index] + ord('A'))}")
    plt.axis("off")
    plt.show()


# Prediction function
def load_and_predict_image(folder, model, img_size=28):
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return None

    # Get all image paths from the folder
    image_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))



    # Select 5 random images
    random_images = random.sample(image_paths, 5)

    for image_path in random_images:
        img = cv2.imread(image_path, 0)
        if img is None:
            print(f"Failed to read image: {image_path}")
            continue

        resized_img = cv2.resize(img, (img_size, img_size))
        normalized_img = resized_img / 255.0
        input_img = np.expand_dims(normalized_img, axis=(0, -1))

        prediction = model.predict(input_img)
        predicted_label = np.argmax(prediction)

        # Show prediction
        plt.imshow(resized_img, cmap='gray')
        plt.title(f"Predicted: {chr(predicted_label + ord('A'))}")
        plt.axis("off")
        plt.show()

# Load dataset
folder = "BigDataSet"  # Adjust this path if needed
X, Y = load_and_preprocess_images(folder, img_size=28)

# Train/test split with stratification
X_train, X_test, Y_train, Y_test = train_test_split(
X, Y, test_size=0.1, random_state=42, stratify=Y)

# Preview sample
show_example(X_train, Y_train, 10)

# Model definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(26, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X_train, Y_train, epochs=20, validation_data=(X_test, Y_test))

# Evaluate
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Save model
model.save("mein_Buchstaben_model.keras")
model.save("mein_Buchstaben_model.h5")
print("Model saved as mein_Buchstaben_model.keras")
print("Model saved as mein_Buchstaben_model.h5")

# Reload model
model = tf.keras.models.load_model("mein_Buchstaben_model.keras")

# Predict 5 random images
load_and_predict_image("BigDataSet", model)