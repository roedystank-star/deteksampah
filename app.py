import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load Model
model = tf.keras.models.load_model("model.h5")

# Load Labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# UI
st.title("♻️ Deteksi Sampah Organik / Anorganik dengan AI")
st.write("Upload gambar sampah untuk dideteksi oleh model AI")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # Preprocessing untuk prediksi
    img = image.resize((224, 224))  # ukuran default TM
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index] * 100

    result = labels[class_index]

    st.subheader("📌 Hasil Deteksi")
    st.write(f"Jenis sampah: **{result}**")
    st.write(f"Akurasi: **{confidence:.2f}%** ✅")
