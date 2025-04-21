import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import time
import cv2
import numpy as np

st.set_page_config(page_title="Детекция каски", page_icon="🦺")


st.title("🦺 Детекция каски с помощью YOLOv8")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.sidebar.markdown("## Источник изображения")
source = st.sidebar.radio("Выберите:", ["Загрузить файл", "Сделать фото с камеры"])

img_path = None
img = None

if source == "Загрузить файл":
    st.markdown("### 📤 Загрузка изображения")
    uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption='Загруженное изображение', use_column_width=True)
        img_path = "temp.jpg"
        img.save(img_path)

elif source == "Сделать фото с камеры":
    st.markdown("### 📸 Сделайте фото")
    photo = st.camera_input("Кликните, чтобы сфотографировать")
    if photo:
        img = Image.open(photo)
        st.image(img, caption='Сделанное фото', use_column_width=True)
        img_path = "temp.jpg"
        img.save(img_path)

if img_path and os.path.exists(img_path):
    if st.button("🚀 Запустить детекцию"):
        progress = st.progress(0)
        for i in range(0, 100, 20):
            time.sleep(0.1)
            progress.progress(i + 20)

        with st.spinner("Распознаём..."):
            results = model(img_path)
            res_plotted = results[0].plot()

            res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

            st.markdown("### 🖼 Результаты")
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Оригинал", use_column_width=True)
            with col2:
                st.image(res_plotted_rgb, caption="Детекция", use_column_width=True)

st.markdown("""
<hr style="margin-top: 50px; border-top: 1px solid #444;" />
<div style='text-align: center; color: gray; font-size: small;'>
    🚧 DS Capstone Project 2025 | Сделано с ❤️ студенткой Арай Жайсанбек
</div>
""", unsafe_allow_html=True)
