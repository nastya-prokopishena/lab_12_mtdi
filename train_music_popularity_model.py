import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Визначення шляху до файлу
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'spotify-2023.csv')

# Завантаження даних
df = pd.read_csv(DATA_PATH, encoding='cp1251', delimiter=',')

# Перетворення числових стовпців
numeric_cols = [
    'artist_count', 'released_year', 'released_month', 'released_day',
    'in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists',
    'in_apple_charts', 'in_deezer_charts', 'bpm',
    'danceability_%', 'valence_%', 'energy_%',
    'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%'
]

# Конвертація текстових стовпців з числами
df['streams'] = pd.to_numeric(df['streams'].str.replace(',', ''), errors='coerce')
df['in_deezer_playlists'] = pd.to_numeric(df['in_deezer_playlists'].str.replace(',', ''), errors='coerce')
df['in_shazam_charts'] = pd.to_numeric(df['in_shazam_charts'].str.replace(',', ''), errors='coerce')

# Кодування категоріальних змінних
le_key = LabelEncoder()
le_mode = LabelEncoder()
df['key_encoded'] = le_key.fit_transform(df['key'].fillna('Unknown'))
df['mode_encoded'] = le_mode.fit_transform(df['mode'])

# Підготовка ознак та цільової змінної
df['is_popular'] = (df['in_spotify_playlists'] > df['in_spotify_playlists'].median()).astype(int)

# Вибираємо ознаки для моделі
features = numeric_cols + ['key_encoded', 'mode_encoded']
X = df[features].fillna(0)
y = df['is_popular']

# Розділення на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабування ознак з MinMaxScaler для кращої масштабованості
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Моделювання з RandomForestClassifier (можливо, він дасть кращі результати)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Оцінка моделі
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Збереження моделі та скалера
model_path = os.path.join(BASE_DIR, 'music_popularity_model.pkl')
joblib.dump({
    'model': model,
    'scaler': scaler,
    'le_key': le_key,
    'le_mode': le_mode,
    'features': features
}, model_path)

print(f"Модель успішно збережена у '{model_path}'")
