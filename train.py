import pandas as pd
import cv2
import os
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

IMG_SIZE = 224

# Load CSV
data = pd.read_csv("dataset/train.csv")

# Load images
images = []
labels = []
print("Loading images...")

for index, row in data.iterrows():
    path = "dataset/train_images/" + row['id_code'] + ".png"

    img = cv2.imread(path)

    if img is None:
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    images.append(img)
    labels.append(row['diagnosis'])

# IMPORTANT CONVERSION
images = np.array(images)
labels = np.array(labels)
print("Images Loaded:", images.shape)

images = np.array(images)
labels = np.array(labels)

print("Images Loaded:", images.shape)

# Train validation split
X_train, X_val, y_train, y_val = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42
)

# MobileNetV2 base
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(128, activation='relu')(x)

predictions = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=4
)

# Save model
model.save("models/dr_model_mobilenet.h5")

print("Model Saved")