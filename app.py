import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import time
import cv2


st.set_page_config(page_title="Проверка каски", page_icon="🦺")


st.title("🛡️ Проверка наличия каски перед входом на производственный участок")


#@st.cache_resource
def load_model():
    return YOLO("best.pt")  

model = load_model()

st.sidebar.markdown("## 🔍 Выберите источник изображения")
source = st.sidebar.radio("Источник:", ["📤 Загрузить файл", "📸 Сделать фото с камеры"])

img_path = None
img = None


if source == "📤 Загрузить файл":
    st.markdown("### 📤 Загрузка изображения работника")
    uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Загруженное изображение", use_container_width=True)
        img_path = "temp.jpg"
        img.save(img_path)

elif source == "📸 Сделать фото с камеры":
    st.markdown("### 📸 Фото сотрудника перед входом")
    photo = st.camera_input("Сделайте фото")
    if photo:
        img = Image.open(photo)
        st.image(img, caption="Сделанное фото", use_container_width=True)
        img_path = "temp.jpg"
        img.save(img_path)

if img_path and os.path.exists(img_path):
    if st.button("✅ Проверить наличие каски"):
        progress = st.progress(0, text="Идёт проверка...")
        for i in range(0, 100, 20):
            time.sleep(0.1)
            progress.progress(i + 20, text="Распознаём изображение...")

        with st.spinner("Проводится анализ..."):
            results = model(img_path)
            res_plotted = results[0].plot()
            res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

            labels = results[0].names
            classes_detected = [results[0].names[int(cls)] for cls in results[0].boxes.cls]

            has_helmet = any("helmet" in c.lower() for c in classes_detected)

            st.markdown("### 🔎 Результаты проверки")
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Исходное изображение", use_container_width=True)
            with col2:
                st.image(res_plotted_rgb, caption="Результат детекции", use_container_width=True)

            st.markdown("---")
            if has_helmet:
                st.success("✅ ДОПУСК РАЗРЕШЁН: Каска обнаружена. Доступ к зоне производства открыт.")
            else:
                st.error("⛔ ДОПУСК ЗАПРЕЩЁН: Каска не обнаружена. Наденьте каску для продолжения.")

st.markdown("""
<hr style="margin-top: 50px; border-top: 1px solid #444;" />
<div style='text-align: center; color: gray; font-size: small;'>
    🦺 DS Capstone Project 2025| Проверка каски перед началом работы | by Арай Жайсанбек
</div>
""", unsafe_allow_html=True)
