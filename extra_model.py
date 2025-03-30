import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

df = pd.read_csv("preprocessed_traffic_data.csv")

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)

sequence_length = 5


def create_sequences(data, time_steps=sequence_length):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps, -1])
    return np.array(X), np.array(y)


X, y = create_sequences(scaled_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(128, return_sequences=True, activation='tanh', input_shape=(X.shape[1], X.shape[2])),
    BatchNormalization(),
    Dropout(0.2),

    LSTM(64, return_sequences=True, activation='tanh'),
    BatchNormalization(),
    Dropout(0.2),

    LSTM(32, return_sequences=False, activation='tanh'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae', metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-5)

epochs = 50
batch_size = 32
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
                    callbacks=[early_stopping, lr_scheduler])

model.save("traffic_lstm_model_optimized.h5")
print("✅ Optimized LSTM Model training completed and saved!")

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss', color='blue', linestyle='--', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linestyle='-', marker='s')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs. Validation Loss (Optimized)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
