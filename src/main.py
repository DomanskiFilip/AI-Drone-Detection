import os
from dotenv import load_dotenv
import kagglehub
from sklearn.model_selection import train_test_split
from models import EuclideanDistance, KNearestNeighbors, MultiLayerPerceptron
from utils import load_and_preprocess_data, evaluate_model, clean_nan_values
from visualise import plot_metrics_over_runs, plot_model_comparison, plot_confusion_matrices
from run_tracker import save_run

# Load variables from .env into the system environment
load_dotenv()

def setup_kaggle():
    # Fetch from environment variables
    username = os.getenv('KAGGLE_USERNAME')
    key = os.getenv('KAGGLE_KEY')
    
    if not username or not key:
        raise ValueError("Kaggle credentials not found in .env file")

    # Configure local Kaggle credentials path
    kaggle_path = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_path, exist_ok=True)
    
    with open(os.path.join(kaggle_path, "kaggle.json"), "w") as f:
        f.write(f'{{"username":"{username}","key":"{key}"}}')
    
    # Secure the file
    if os.name != 'nt': # If not Windows
        os.chmod(os.path.join(kaggle_path, "kaggle.json"), 0o600)

def main():
    # Download/Path Setup
    path = kagglehub.dataset_download("maryamlsgumel/drone-detection-dataset")
    base_path = f"{path}/BirdVsDroneVsAirplane"
    
    categories = {'Birds': 0, 'Drones': 1, 'Aeroplanes': 2}
    class_names = list(categories.keys())
    
    # Load Data
    print("Loading data...")
    data, labels = load_and_preprocess_data(base_path, categories)
    
    # Clean NaN values if any (though unlikely in image data, this is just a good practice step)
    print("Cleaning NaN values...")
    data = clean_nan_values(data)
    labels = clean_nan_values(labels).astype(int)
    
    # Flatten and Split
    n_samples = data.shape[0]
    features = data.reshape(n_samples, -1)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Run Models
    models = [
        EuclideanDistance(),
        KNearestNeighbors(k=5),
        MultiLayerPerceptron(hidden_neurons=100, epochs=30)
    ]

    for model in models:
        print(f"\nTraining {model.name}...")
        model.train(X_train, y_train)
        results = evaluate_model(model, X_test, y_test, class_names)
        save_run(results, notes="baseline run")

if __name__ == "__main__":
    main()