import os
import cv2
import numpy

def load_and_preprocess_data(base_path, categories, img_size=(64, 64)):
    data, labels = [], []
    for folder_name, label in categories.items():
        category_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(category_path):
            continue
        
        for img_name in os.listdir(category_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                data.append(img.astype('float32') / 255.0)
                labels.append(label)
                
    return numpy.array(data), numpy.array(labels)

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = (numpy.sum(predictions == y_test) / len(y_test)) * 100
    print(f"{model.name} Accuracy: {accuracy:.2f}%")
    return accuracy