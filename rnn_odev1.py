import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
# %%
# 1. Veri Seti Oluşturma (Sentetik Sinüs Dalgası)
def create_sine_wave(n_points=2000):
    time = np.linspace(0, 100, n_points)
    series = np.sin(time) + np.random.normal(0, 0.1, n_points) # Biraz gürültü ekledik
    return series

series = create_sine_wave()

# %%
# 2. Veriyi Pencereleme (Windowing)
# RNN'ler (L, N) formatında veri ister: [Örnek Sayısı, Zaman Adımı, Özellik Sayısı]
def windowed_dataset(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

# %%
window_size = 50 # Geçmiş 50 güne bak
X, y = windowed_dataset(series, window_size)

# Boyutları düzenleme: (Örnek, Zaman Adımı, Özellik Sayısı)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Eğitim ve Test ayırımı
split = int(0.8 * len(X))
x_train, x_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# %%
# 3. Model Kurulumu (SimpleRNN ve GRU Örneği)
# GRU, LSTM'in daha hızlı çalışan ve benzer performans veren bir türevidir.
model = models.Sequential([
    layers.Input(shape=(window_size, 1)),
    layers.SimpleRNN(64, activation='tanh', return_sequences=True), # Katmanları üst üste koyarken return_sequences=True
    layers.SimpleRNN(32),
    layers.Dense(1) # Gelecek tek bir değeri tahmin ediyoruz
])
# %%
# 4. Derleme
model.compile(optimizer='adam', loss='mse') # Regresyon olduğu için Mean Squared Error

# %%
# 5. Eğitim
history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(x_test, y_test),
    verbose=1
)
# %%
# 6. Tahmin ve Görselleştirme
predictions = model.predict(x_test)

plt.figure(figsize=(12, 4))
plt.plot(y_test[:200], label="Gerçek Değerler")
plt.plot(predictions[:200], label="RNN Tahminleri")
plt.legend()
plt.title("Zaman Serisi Tahmini")
plt.show()