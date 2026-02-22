import os
from dotenv import load_dotenv
import kagglehub
from sklearn.model_selection import train_test_split
from models import EuclideanDistance, KNearestNeighbors, MultiLayerPerceptron
from utils import load_and_preprocess_data, evaluate_model

# Load variables from .env into the system environment
load_dotenv()

def setup_kaggle():
    # Fetch from environment variables
    username = os.getenv('KAGGLE_USERNAME')
    key = os.getenv('KAGGLE_KEY')
    
    if not username or not key:
        raise ValueError("Kaggle credentials not found in .env file")

    # Configure local Kaggle credentials path
    # This mimics what you did in Colab but for your local PC
    kaggle_path = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_path, exist_ok=True)
    
    with open(os.path.join(kaggle_path, "kaggle.json"), "w") as f:
        f.write(f'{{"username":"{username}","key":"{key}"}}')
    
    # Secure the file (standard Kaggle requirement)
    if os.name != 'nt': # If not Windows
        os.chmod(os.path.join(kaggle_path, "kaggle.json"), 0o600)

def main():
    # 1. Download/Path Setup
    path = kagglehub.dataset_download("maryamlsgumel/drone-detection-dataset")
    base_path = f"{path}/BirdVsDroneVsAirplane"
    
    categories = {'Birds': 0, 'Drones': 1, 'Aeroplanes': 2}
    
    # 2. Load Data
    print("Loading data...")
    data, labels = load_and_preprocess_data(base_path, categories)
    
    # 3. Flatten and Split
    n_samples = data.shape[0]
    features = data.reshape(n_samples, -1)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 4. Run Models
    models = [
        EuclideanDistance(),
        KNearestNeighbors(k=5),
        MultiLayerPerceptron(hidden_neurons=100, epochs=30)
    ]

    for model in models:
        print(f"\nTraining {model.name}...")
        model.train(X_train, y_train)
        evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()