import joblib
import pandas as pd
import numpy as np
import sys
import os


def load_model(model_path='music_popularity_model.pkl'):
    return joblib.load(model_path)


def predict_track_popularity(input_data, model_data):
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']

    input_scaled = scaler.transform(input_data[features])

    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)

    return prediction, probabilities


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'music_popularity_model.pkl')

    model_data = load_model(model_path)

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        input_df = pd.read_csv(input_path, encoding='cp1251', delimiter=',')
    else:
        print("Будь ласка, вкажіть файл CSV з вхідними даними.")
        sys.exit(1)

    predictions, probabilities = predict_track_popularity(input_df, model_data)

    input_df['predicted_popularity'] = predictions
    input_df['popularity_probability'] = probabilities[:, 1]

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    print("\nРезультати передбачення:")
    print(input_df[['track_name', 'artist(s)_name','predicted_popularity', 'popularity_probability']].to_string())


if __name__ == "__main__":
    main()