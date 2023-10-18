import tensorflow as tf

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
    data_augmentation,  # Augments/Add Variation to Input
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), # Extracts Features
    tf.keras.layers.MaxPooling2D((2, 2)), # Downscales Feature Map
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), # Extracts Higher Level Features
    tf.keras.layers.MaxPooling2D((2, 2)), # Further Down Scales Feature Map
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'), # Extracts Higher Level Features
    tf.keras.layers.MaxPooling2D((2, 2)), # Further Down Scales Feature Map
    tf.keras.layers.Flatten(), # Converts the 2 dimensional Feature Map into a 1 Dimensional Feature Vector
    tf.keras.layers.Dense(128, activation='relu'), # Processes The Features from Feature Vector, 128 "Switches"
    tf.keras.layers.Dense(1, activation='sigmoid')  # Processes Output with 1 "Switch" indicating which of two classes for each image
])

model.compile(optimizer='adam', # Adam is the selected optimization algorithm, recommended idk
              loss='binary_crossentropy',  # Loss function used, problem is binary classification so... 
              metrics=['accuracy']) # Measures quality of model by percentage of correctly identified.


