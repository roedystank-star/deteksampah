import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow.lite as tflite
import os

st.set_page_config(page_title="Deteksi Sampah (TFLite)", layout="centered")
st.title("♻️ Deteksi Sampah Organik vs Anorganik (TFLite)")
st.write("Ambil foto menggunakan kamera HP atau upload gambar.")

MODEL_PATH = "model.tflite"
LABELS_PATH = "labels.txt"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model tidak ditemukan: {MODEL_PATH}. Pastikan file ada di folder proyek.")
    st.stop()

if not os.path.exists(LABELS_PATH):
    st.error(f"Labels tidak ditemukan: {LABELS_PATH}.")
    st.stop()

# Muat label
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines() if line.strip()]

# Muat interpreter
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Ambil input shape info
in_shape = input_details[0]['shape']  # e.g. [1,224,224,3]
# handle dynamic dims: set H,W
_, H, W, C = in_shape if len(in_shape) == 4 else (1,224,224,3)

def preprocess_image(img: Image.Image):
    # Pastikan mode RGB
    img = img.convert("RGB")
    # resize mempertahankan aspect ratio -> fill (opsional) atau langsung resize
    img = ImageOps.fit(img, (W, H), Image.ANTIALIAS)
    arr = np.array(img).astype(np.float32)
    # Normalisasi sesuai Teachable Machine (biasanya /255.0)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# Input (kamera atau upload)
img_file = st.camera_input("Ambil foto sampah (kamera)") or st.file_uploader("Atau upload gambar", type=["jpg","jpeg","png"])

if img_file:
    image = Image.open(img_file)
    st.image(image, caption="Gambar yang diuji", use_column_width=True)

    # Preprocess
    input_data = preprocess_image(image)

    # Set input (beberapa model mengharapkan dtype float32)
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(input_details[0]['dtype']))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Untuk model berbentuk [1, N] probabilities
    probs = np.squeeze(output_data)
    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])

    label_text = labels[top_idx] if top_idx < len(labels) else f"kelas_{top_idx}"
    st.success(f"Hasil: **{label_text.capitalize()}** ({confidence*100:.2f}%)")

    # Tampilkan semua probabilitas
    st.write("Probabilitas tiap kelas:")
    for i, lab in enumerate(labels):
        st.write(f"- {lab.capitalize()}: {probs[i]*100:.2f}%")
