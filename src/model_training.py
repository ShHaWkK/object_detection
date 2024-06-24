import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from src.data_preprocessing import preprocess_image
import os

def load_data(data_dir, target_size):
    X, y = [], []
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                image = preprocess_image(img_path, target_size)
                X.append(image)
                y.append(class_dir)  
    X = np.vstack(X)
    y = np.array(y)
    return X, y

def train_model():
    target_size = (224, 224)
    data_dir = 'data/images/train'
    X_train, y_train = load_data(data_dir, target_size)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(set(y_train)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10)

    model.save('models/trained_model/object_detection_model.h5')
    print("Model training complete.")
