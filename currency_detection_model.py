import tensorflow as tf
import matplotlib.pyplot as plt

# Paths
test_path = 'Indian_Currency/Test'
train_path = 'Indian_Currency/Train'

# Image parameters
img_size = (224, 224)
batch_size = 32

# Load raw datasets
train_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    image_size=img_size,
    batch_size=batch_size
)

test_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    image_size=img_size,
    batch_size=batch_size
)

# ✅ Save class names BEFORE normalization
class_names = train_ds_raw.class_names

# ✅ Normalize images to 0–1
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds_raw.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds_raw.map(lambda x, y: (normalization_layer(x), y))

# Build MobileNetV2-based model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_ds, validation_data=test_ds, epochs=10)

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show(block=False)
plt.pause(3)
plt.close()

# Final accuracy
val_acc = history.history['val_accuracy'][-1]
print(f"Final Validation Accuracy: {val_acc * 100:.2f}%")

# Show class names
print("Class Names:", class_names)

# Save model
model.save("currency_detection.h5")
