import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.title("♻️ Deteksi Sampah Organik & Anorganik")
st.write("Gunakan kamera HP atau upload gambar untuk mendeteksi jenis sampah.")

# Cek apakah model tersedia
model_path = "model_sampah"
if not os.path.exists(model_path):
    st.error("❌ Folder model_sampah tidak ditemukan. Pastikan folder model sudah diunggah ke GitHub.")
else:
    # Muat model
    model = tf.keras.models.load_model(model_path)
    st.success("✅ Model berhasil dimuat!")

    # Input gambar
    img_camera = st.camera_input("Ambil foto sampah (gunakan kamera HP)")
    uploaded_file = st.file_uploader("Atau upload gambar", type=["jpg", "jpeg", "png"])

    image = None
    if img_camera is not None:
        image = Image.open(img_camera)
    elif uploaded_file is not None:
        image = Image.open(uploaded_file)

    if image:
        st.image(image, caption="Gambar yang diuji", use_column_width=True)

        # Preprocessing
        img = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        # Prediksi
        prediction = model.predict(img_array)
        label = ['Organik', 'Anorganik'][np.argmax(prediction)]
        prob = float(np.max(prediction))

        st.info(f"Hasil: **{label}** (kepastian {prob*100:.2f}%)")

        # Pesan tambahan
        if label == 'Organik':
            st.success("💡 Sampah organik bisa dijadikan kompos!")
        else:
            st.warning("♻️ Sampah anorganik sebaiknya dipilah dan didaur ulang.")
