import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.data_preprocessing import preprocess_image

def detect_objects():
    model_path = 'models/trained_model/object_detection_model.h5'
    model = load_model(model_path)

    image_path = 'data/images/test/test_image.jpg' 
    target_size = (224, 224)
    image = preprocess_image(image_path, target_size)

    predictions = model.predict(image)
    class_idx = np.argmax(predictions)
    confidence = np.max(predictions)

    print(f"Detected class: {class_idx} with confidence: {confidence}")

    # Afficher l'image avec la classe détectée
    image = cv2.imread(image_path)
    cv2.putText(image, f"Class: {class_idx}, Confidence: {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Object Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
