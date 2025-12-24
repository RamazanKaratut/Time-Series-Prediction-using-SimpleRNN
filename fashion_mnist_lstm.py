import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# %%
# 1. Load and Preprocess Dataset
# Fashion-MNIST: 70,000 grayscale images in 10 categories
print("Loading Fashion-MNIST data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Each image is 28x28 pixels. 
# We will treat each row (28) as a time step and each pixel in the row (28) as a feature.
# Shape: (Samples, Time Steps, Features) -> (Batch, 28, 28)

# %%
# 2. Build LSTM Model
model = models.Sequential([
    layers.Input(shape=(28, 28)),
    layers.LSTM(128, activation='tanh', return_sequences=True),
    layers.LSTM(64, activation='tanh'),
    layers.Dropout(0.2), # Regularization to prevent overfitting
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax') # 10 classes for Fashion-MNIST
])

# %%
# 3. Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# %%
# 4. Train Model
print("Starting training...")
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# %%
# 5. Evaluate and Visualize
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest Accuracy: {test_acc*100:.2f}%')

# Plot training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# %%
# 6. Make Predictions
predictions = model.predict(x_test)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualizing a single prediction
plt.figure()
plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.title(f"Actual: {class_names[y_test[0]]}\nPredicted: {class_names[np.argmax(predictions[0])]}")
plt.axis('off')
plt.show()
