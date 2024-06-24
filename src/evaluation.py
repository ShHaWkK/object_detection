from tensorflow.keras.models import load_model
from src.data_preprocessing import preprocess_image
import numpy as np
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

def evaluate_model():
    model_path = 'models/trained_model/object_detection_model.h5'
    model = load_model(model_path)

    data_dir = 'data/images/test'
    target_size = (224, 224)
    X_test, y_test = load_data(data_dir, target_size)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Model evaluation complete. Loss: {loss}, Accuracy: {accuracy}")
