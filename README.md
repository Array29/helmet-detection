# 🟡 Проект: Детекция касок с помощью YOLOv8

## 📌 Описание задачи

Проект представляет собой интеллектуальную систему компьютерного зрения на базе YOLOv8, способную определять наличие защитных касок у сотрудников. Такое решение не только усиливает меры техники безопасности, но и становится важным элементом в инфраструктуре индустрии 4.0 — умного, автономного и безопасного производства.

🔍 Назначение системы:
Система может быть внедрена в различные производственные сценарии:

🏭 Роботизированные линии — блокировка запуска оборудования при отсутствии СИЗ (средств индивидуальной защиты).

🤖 Мобильные и стационарные роботы — контроль производственных зон и автоматическое реагирование на нарушения.

🔐 Цифровые системы доступа — запрет входа в опасные зоны без каски, исключая человеческий фактор.

Интеграция модели в такие процессы позволяет создать автономную экосистему безопасности, где контроль осуществляется алгоритмами, а не людьми.
## 💼 Business Value
На промышленных предприятиях нарушения техники безопасности ведут к травмам, штрафам и простоям оборудования. Этот проект — шаг к интеллектуальной автоматизации охраны труда, где ответственность за безопасность несут не охранники, а технологии.

Ключевые преимущества:

🔧 Безопасность по умолчанию: система распознаёт сотрудников без СИЗ, исключая риски.

🤖 Интеллектуальный контроль допуска: запуск операций возможен только при соблюдении требований безопасности.

🏭 Минимизация простоев и рисков: устранение человеческой ошибки снижает аварийность и потери.

📊 Мониторинг и аналитика: визуализация нарушений, выявление проблемных зон и построение отчётов.

🚀 Гибкая интеграция: применяется в AMR, манипуляторах, сварочных и сборочных системах.

Итог:
Проект — это прототип цифрового охранника, встроенного в производственные процессы, работающего без усталости 24/7. Он помогает создавать саморегулирующиеся, безопасные и эффективные производственные среды нового поколения.

## 🗂 Описание датасета

- Формат: YOLOv8
- Классы:
  - `helmet` — человек в каске
  - `head` — человек без каски
- Структура:
  - `images/train`, `images/val`, `images/test` — изображения
  - `labels/train`, `labels/val`, `labels/test` — соответствующие аннотации
- Источник: [Kaggle Hard Hat Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)

## ⚙️ Технические характеристики

- **Архитектура:** YOLOv8

- **Формат модели:** `.pt` (PyTorch checkpoint)

- **Фреймворк:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  

- **Язык разработки:** Python

## 🔧 Гиперпараметры при обучении

- `imgsz=640` — размер входного изображения  
- `epochs=50-100` — количество эпох обучения  
- `batch=16` — размер батча  
- **Аугментации включены**:
  - Отражение (`flip`)
  - Масштабирование (`scale`)
  - Смещение по оттенку/насыщенности/яркости (`HSV shift`)
  - Прочие встроенные аугментации Ultralytics YOLOv8

augmentation_params = {
    "degrees": 10,
    "translate": 0.2,
    "scale": 0.3,
    "shear": 2,
    "perspective": 0.0005,
    "flipud": 0.3,
    "fliplr": 0.5,
    "mosaic": 1,
    "mixup": 0.1,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4
} 

## 📊 Метрики модели

| Метрика           | Значение |
|-------------------|----------|
| mAP@0.5           | 0.938    |
| mAP@0.5:0.95      | 0.622    |
| Precision         | 0.925    |
| Recall            | 0.868    |

## 🖥 Применение в Streamlit-приложении

- **Загрузка модели**:

### Входные данные: 
изображение пользователя (с камеры или загруженное из файла)

### Выход: 
изображение с размеченными объектами (helmet, head)

### Интерфейс: 
приложение реализовано с использованием Streamlit Cloud и предоставляет простой и интуитивно понятный пользовательский интерфейс. Пользователь может загрузить свое фото либо же сделать фото онлайн. Система определяет наличие каски и на основе этого либо разрешает, либо запрещает допуск на производство. 
Ссылка на проект в [Streamlit Cloud](https://helmet-detection-zs7hcnlsnkqw8evyzwnz7d.streamlit.app/) 
