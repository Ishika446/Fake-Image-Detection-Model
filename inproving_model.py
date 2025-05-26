import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 📂 Paths
data_dir = r'C:\Users\Ishika\Fake image detector\valid'
model_path = r'C:\Users\Ishika\Fake image detector\fake_vs_real_model(1).keras'

# ⚙️ Hyperparameters
batch_size = 32
img_size = (128, 128)

# 📦 Data loading
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.3,
    shear_range=0.3,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)


train_data = datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', subset='training')
val_data = datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size, class_mode='binary', subset='validation')

# ✅ Recreate model exactly as before
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
base_model.trainable = False  # Freeze for now

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

# Load previous weights
model.load_weights(model_path)
print("✅ Model loaded.")

# 🔓 Fine-tuning: Unfreeze base model
base_model.trainable = True

# Freeze first N layers, fine-tune last 50
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Re-compile with a lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# 📈 Fine-tune
history = model.fit(train_data, validation_data=val_data, epochs=10)

def plot_and_save_training_history(history, save_dir="plots"):
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the combined plot
    save_path = r"C:\Users\Ishika\Fake image detector\training_history(1).png"
    plt.savefig(save_path)
    print(f"✅ Training graph saved to: {save_path}")

    plt.show()

plot_and_save_training_history(history)
# 💾 Save updated model
model.save(r"C:\Users\Ishika\Fake image detector\fake_vs_real_model(2).keras")
