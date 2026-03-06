import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 224

model = load_model("models/dr_model_mobilenet.h5")

img = cv2.imread("test.png")

img = cv2.resize(img,(224,224))
img = img/255.0

img = np.expand_dims(img, axis=0)

prediction = model.predict(img)

class_id = np.argmax(prediction)

classes = [
"No DR",
"Mild",
"Moderate",
"Severe",
"Proliferative"
]

confidence = np.max(prediction)
print("Prediction:", classes[class_id])
print("Confidence:", round(confidence*100,2), "%")
