import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

st.set_page_config(page_title="Детекция каски", page_icon="🦺")
st.title("🦺 Детекция каски с помощью YOLOv8")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

source = st.radio("Выберите источник изображения:", ["Загрузить файл", "Сделать фото с камеры"])

if source == "Загрузить файл":
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption='Загруженное изображение', use_column_width=True)
        img_path = "temp.jpg"
        img.save(img_path)

elif source == "Сделать фото с камеры":
    photo = st.camera_input("Сделайте фото")
    if photo:
        img = Image.open(photo)
        st.image(img, caption='Сделанное фото', use_column_width=True)
        img_path = "temp.jpg"
        img.save(img_path)

if ('img_path' in locals()) and os.path.exists(img_path):
    with st.spinner("Распознаём..."):
        results = model(img_path)
        res_plotted = results[0].plot()
        st.image(res_plotted, caption='Результат детекции', use_column_width=True)
