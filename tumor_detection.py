import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

model_path = "D:\\sem3\\final projects\\Brain tumor prediction\Model2\\brain_tumor_model2.h5"
loaded_model = tf.keras.models.load_model(model_path)


def predict_tumor_class(image):

    img = np.array(image)
    img = cv2.resize(img, (150, 150))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0


    prediction = loaded_model.predict(img)

    # Map class indices to class labels
    class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
    predicted_class = class_labels[np.argmax(prediction)]


    decimal_probabilities = [round(prob, 5) for prob in prediction[0]]

    return predicted_class, decimal_probabilities


iface = gr.Interface(
    fn=predict_tumor_class,
    inputs=gr.inputs.Image(type='pil', label="Upload a brain tumor image"),
    outputs=["text", "text"],
    output_names=["Predicted Tumor", "Probability"],
    live=True,
    title="Brain Tumor Detector"
)


iface.launch(share=True)
