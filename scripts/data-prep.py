import tensorflow as tf
import matplotlib.pyplot as plt




train_data = tf.keras.utils.image_dataset_from_directory(
'./data/train',
image_size=(224, 224),
batch_size=32,
)


validation_data = tf.keras.utils.image_dataset_from_directory(
'./data/test',
image_size=(224, 224),
batch_size=32,
)


class_names = train_data.class_names
print(class_names)


data_augmentation = tf.keras.Sequential([
tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
])


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)), # Specifies Input Format
    data_augmentation, # Augments/Add Variation to Input
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), # Extracts Features
    tf.keras.layers.MaxPooling2D((2, 2)), # Downscales Feature Map
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), # Extracts Higher Level Features
    tf.keras.layers.MaxPooling2D((2, 2)), # Further Down Scales Feature Map
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'), # Extracts Higher Level Features
    tf.keras.layers.MaxPooling2D((2, 2)), # Further Down Scales Feature Map
    tf.keras.layers.Flatten(), # Converts the 2 dimensional Feature Map into a 1 Dimensional Feature Vector
    tf.keras.layers.Dense(128, activation='relu'), # Processes The Features from Feature Vector, 128 "Switches"
    tf.keras.layers.Dense(1, activation='sigmoid') # Processes Output with 1 "Switch" indicating which of two classes for each image
])


model.compile(optimizer='adam', # Adam is the selected optimization algorithm, recommended idk
loss='binary_crossentropy', # Loss function used, problem is binary classification so...
metrics=['accuracy']) # Measures quality of model by percentage of correctly identified.


model.summary()


history = model.fit(
train_data,
validation_data=validation_data,
epochs=10
)


model.save("predict_model.h5")




## Prints MetaData
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']


loss = history.history['loss']
val_loss = history.history['val_loss']


epochs_range = range(10)


plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')


plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
