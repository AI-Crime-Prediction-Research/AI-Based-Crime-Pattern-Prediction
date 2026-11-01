import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load pretrained CNN (no training needed)
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

frame_folder = "data/video_data/frames"

features = []

print("Processing frames...")

for img_name in os.listdir(frame_folder):
    img_path = os.path.join(frame_folder, img_name)

    try:
        img = image.load_img(img_path, target_size=(224,224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        feature = model.predict(img_array, verbose=0)
        features.append(feature.flatten())

    except Exception as e:
        print(f"Skipping {img_name}: {e}")

features = np.array(features)

print("Feature shape:", features.shape)
np.save("data/video_data/cnn_features.npy", features)
print("Feature shape:") , features.shape
print("Feature shape:" , features.shape)
np.save("data/video_data/cnn")