from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI()

model = tf.keras.applications.MobileNetV2(weights="imagenet")

def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/")
def home():
    return {"message": "CNN working"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    processed = preprocess(image)

    preds = model.predict(processed)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)

    return {"prediction": decoded[0][0][1]}