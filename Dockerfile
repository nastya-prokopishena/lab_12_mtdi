# Вибір базового образу
FROM python:3.11-slim

# Встановлення необхідних бібліотек
RUN pip install --no-cache-dir pandas numpy scikit-learn joblib

# Копіювання файлів у контейнер
COPY . /app

# Встановлення робочої директорії
WORKDIR /app

# Запуск програми
CMD ["python", "predict_music_popularity.py"]
