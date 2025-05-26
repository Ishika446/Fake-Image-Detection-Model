import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# 🔧 Paths
data_dir = r'C:\Users\Ishika\Fake image detector\data'  # path to folder containing 'real' and 'fake'
batch_size = 32
img_size = (128, 128)

# 📦 Data loading
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# 🧠 Transfer Learning: MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

# ⚙️ Compile with lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 🏋️ Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)

# 💾 Save model
model.save(r"C:\Users\Ishika\Fake image detector\fake_vs_real_model.keras")

# ✅ Check that the model works
sample_img_path = val_data.filepaths[0]
img = tf.keras.utils.load_img(sample_img_path, target_size=img_size)
img_array = tf.keras.utils.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]
label = 'real' if prediction > 0.5 else 'fake'
confidence = round(float(prediction if label == 'real' else 1 - prediction), 4)

print(f"📷 Image: {sample_img_path}")
print(f"🔍 Prediction: {label} (Confidence: {confidence})")
