import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Завантаження тестових даних
test_data_1 = pd.read_csv('test_data_1.csv', encoding='cp1251', delimiter=',')
test_data_2 = pd.read_csv('test_data_2.csv', encoding='cp1251', delimiter=',')
test_data_3 = pd.read_csv('test_data_3.csv', encoding='cp1251', delimiter=',')

# Завантажуємо тренувальні дані (якщо вони є) для кодування
train_data = pd.read_csv('spotify-2023.csv', encoding='cp1251', delimiter=',')  # припустимо, що є тренувальні дані

# Кодування категоріальних змінних
le_key = LabelEncoder()
le_mode = LabelEncoder()

# Об'єднуємо тренувальні та тестові дані для кодування
combined_key = pd.concat([train_data['key'], test_data_1['key'], test_data_2['key'], test_data_3['key']], axis=0)
combined_mode = pd.concat([train_data['mode'], test_data_1['mode'], test_data_2['mode'], test_data_3['mode']], axis=0)

# Навчаємо LabelEncoder на комбінованих даних
le_key.fit(combined_key.fillna('Unknown'))
le_mode.fit(combined_mode.fillna('Unknown'))

# Тепер можна застосувати трансформацію до кожного тестового набору
test_data_1['key_encoded'] = le_key.transform(test_data_1['key'].fillna('Unknown'))
test_data_1['mode_encoded'] = le_mode.transform(test_data_1['mode'].fillna('Unknown'))

test_data_2['key_encoded'] = le_key.transform(test_data_2['key'].fillna('Unknown'))
test_data_2['mode_encoded'] = le_mode.transform(test_data_2['mode'].fillna('Unknown'))

test_data_3['key_encoded'] = le_key.transform(test_data_3['key'].fillna('Unknown'))
test_data_3['mode_encoded'] = le_mode.transform(test_data_3['mode'].fillna('Unknown'))

# Перезаписуємо файли з оновленими даними
test_data_1.to_csv('test_data_1.csv', index=False)
test_data_2.to_csv('test_data_2.csv', index=False)
test_data_3.to_csv('test_data_3.csv', index=False)
