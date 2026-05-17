import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, GaussianNoise
from sklearn.utils import class_weight
import json
import os

# Load 70-feature data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
with open("label_map.json", "r") as f:
    label_map = json.load(f)

num_classes = len(np.unique(y_train))
y_cat = tf.keras.utils.to_categorical(y_train, num_classes)

# Focal Weighting for the Research Cluster
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights_dict = {i: w for i, w in enumerate(weights)}
for char in ['R', 'U', 'W', 'Y', 'M', 'N', 'T']:
    if char in label_map:
        weights_dict[label_map[char]] *= 8.0 

model = tf.keras.Sequential([
    Input(shape=(70,)), # Updated to 70
    GaussianNoise(0.01),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_cat, epochs=150, batch_size=32, class_weight=weights_dict, validation_split=0.2)

if not os.path.exists("model"): os.makedirs("model")
model.save("model/sign_model.h5")
print("✅ 70-Feature Model Saved.")