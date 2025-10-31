import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("♻️ Deteksi Sampah Organik & Anorganik")
st.write("Gunakan kamera HP atau upload gambar sampah.")

# Gunakan kamera langsung
img_camera = st.camera_input("Ambil foto sampah (gunakan kamera HP)")

# Atau upload dari file
uploaded_file = st.file_uploader("Atau upload gambar sampah", type=["jpg", "jpeg", "png"])

image = None
if img_camera is not None:
    image = Image.open(img_camera)
elif uploaded_file is not None:
    image = Image.open(uploaded_file)

if image:
    st.image(image, caption="Gambar yang diuji", use_column_width=True)

    # Muat model
    model = tf.keras.models.load_model("model_sampah")

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Prediksi
    prediction = model.predict(img_array)
    label = ['Organik', 'Anorganik'][np.argmax(prediction)]

    st.success(f"✅ Jenis sampah terdeteksi: **{label}**")
