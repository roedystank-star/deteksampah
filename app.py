import streamlit as st
import numpy as np
from PIL import Image
import tensorflow.lite as tflite

st.title("♻️ Deteksi Sampah Organik & Anorganik (TFLite Version)")
st.write("Gunakan kamera HP atau upload foto untuk mendeteksi jenis sampah.")

# Muat model tflite
interpreter = tflite.Interpreter(model_path="model_sampah.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img_camera = st.camera_input("Ambil foto sampah")
uploaded_file = st.file_uploader("Atau upload gambar", type=["jpg", "jpeg", "png"])

image = None
if img_camera is not None:
    image = Image.open(img_camera)
elif uploaded_file is not None:
    image = Image.open(uploaded_file)

if image:
    st.image(image, caption="Gambar yang diuji", use_column_width=True)
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    label = ['Organik', 'Anorganik'][np.argmax(prediction)]
    prob = float(np.max(prediction))

    st.success(f"Hasil: {label} ({prob*100:.2f}%)")
