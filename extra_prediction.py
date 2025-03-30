import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate


def mae(y_true, y_pred):
    return tf.keras.losses.mean_absolute_error(y_true, y_pred)


df = pd.read_csv("preprocessed_traffic_data.csv")

df = df[['road_id', 'time_seconds', 'congestion_level', 'avg_speed', 'vehicle_count']]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)

sequence_length = 5
num_features = scaled_data.shape[
    1]


def create_sequences(data, time_steps=sequence_length):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
    return np.array(X)


model = tf.keras.models.load_model("traffic_lstm_model_optimized.h5", custom_objects={"mae": mae})

road_ids = df["road_id"].unique()
road_predictions = []

for road in road_ids:

    road_data = df[df["road_id"] == road].values

    road_scaled = scaler.transform(road_data)

    X_sequences = create_sequences(road_scaled)

    if len(X_sequences) == 0:
        continue

    X_input = X_sequences[-1:].copy()

    future_predictions = []
    current_input = X_input.copy()

    for step in range(5):
        prediction = model.predict(current_input, verbose=0)
        future_predictions.append(prediction[0, 0])

        new_input = np.roll(current_input, shift=-1, axis=1)
        new_input[0, -1, 2] = prediction[0, 0]
        current_input = new_input.copy()

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    dummy_data = np.zeros((5, df.shape[1] - 1))
    full_future_data = np.hstack((dummy_data, future_predictions))
    future_predictions_actual = scaler.inverse_transform(full_future_data)[:, -1]

    road_predictions.append([road] + list(future_predictions_actual))

headers = ["Road ID", "Step 1", "Step 2", "Step 3", "Step 4", "Step 5"]
print("\n🚦 Predicted Congestion Levels for Each Road Over Next 5 Steps:\n")
print(tabulate(road_predictions, headers=headers, tablefmt="grid"))
