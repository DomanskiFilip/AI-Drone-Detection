import os
import cv2
import numpy
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report
)
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

def evaluate_model(model, X_test, y_test, class_names=None):
    predictions = model.predict(X_test)
    accuracy = (numpy.sum(predictions == y_test) / len(y_test)) * 100

    # Precision: correctly predicted / the total predicted
    # High precision means fewer false positives.
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    # Recall:  correctly predicted / all observations in actual class
    # High recall means fewer false negatives.
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    # F1-Score: The weighted average of Precision and Recall. It tries to find the balance between precision and recall
    # A high F1-score indicates good performance for both precision and recall
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    # Confusion Matrix: A table that shows the counts of true positives, true negatives, false positives, and false negatives
    cm = confusion_matrix(y_test, predictions)
    # Mean Absolute Error (MAE): The average of the absolute differences between predictions and actual values
    # It gives an idea of the average magnitude of errors.
    mae = mean_absolute_error(y_test, predictions)
    # Root Mean Squared Error (RMSE): The square root of the average of the squared differences between predictions and actual values
    # average distance from correct predicton
    rmse = numpy.sqrt(mean_squared_error(y_test, predictions))
    # R-squared (R2 Score): Represents the proportion of the variance in the dependent variable that is predictable
    # 1 means model understands the class perfectly 0 means model is no better than random guessing - means model does not capture any of the variance in the data
    r2 = r2_score(y_test, predictions)

    results = {
        "model":     model.name,
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "mae":       round(mae, 4),
        "rmse":      round(rmse, 4),
        "r2":        round(r2, 4),
        "confusion_matrix": cm.tolist()
    }

    print(f"\n{'='*50}")
    print(f"Results for: {model.name}")
    print(f"  Accuracy:  {accuracy:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  MAE:       {mae:.4f}")
    print(f"  RMSE:      {rmse:.4f}")
    print(f"  R²:        {r2:.4f}")
    if class_names:
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=class_names))
    print(f"Confusion Matrix:\n{cm}")

    return results

def clean_nan_values(data_input):
  nan_representations = ['--', 'na', 'n/a', 'nan', 'none', '']

  data_array = numpy.asarray(data_input, dtype=object)

  cleaned_array = numpy.copy(data_array)

  # Iterate through each element of the array
  it = numpy.nditer(data_array, flags=['multi_index', 'refs_ok'])
  for x in it:
      idx = it.multi_index
      value = x.item() # Get the actual value from array

  if isinstance(value, str):
      # If the value is a string, check if its lowercase version matches any NaN representation
      if value.lower() in nan_representations_lower:
          cleaned_array[idx] = numpy.nan
  elif value is None:
      # Explicitly treat Python's None as a NaN
      cleaned_array[idx] = numpy.nan

  # Attempt to convert the cleaned array to a floating-point numeric type
  try:
      temp_numeric_array = cleaned_array.astype(float)
      cleaned_array = temp_numeric_array
  except ValueError:
      print("Warning: Some values could not be converted to numeric type after NaN cleaning. "
            "The array will remain of object dtype to preserve data integrity.")
      pass

  return cleaned_array
