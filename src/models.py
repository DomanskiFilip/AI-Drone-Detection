import numpy
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    SGDClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# defines sklearn based algorythms to be used in the main.py for training and evaluation


# Algorithm 1: Euclidean Distance
class EuclideanDistance:
    def __init__(self):
        self.model = KNeighborsClassifier(
            n_neighbors=1, algorithm="brute", metric="euclidean"
        )
        self.name = "Euclidean Distance"

    def train(self, X_train, y_train):
        print(f"\nTraining {self.name}...")
        self.model.fit(X_train, y_train)
        print(f"Memorized {len(X_train)} training examples")

    def predict(self, X_test):
        return self.model.predict(X_test)


# Algorithm 2: K Nearest Neighbors
class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.model = KNeighborsClassifier(
            n_neighbors=k, algorithm="auto", metric="euclidean"
        )
        self.name = f"K-Nearest Neighbors k={k}"

    def train(self, X_train, y_train):
        print(f"\nTraining {self.name}...")
        self.model.fit(X_train, y_train)
        print(f"Memorized {len(X_train)} training examples")

    def predict(self, X_test):
        return self.model.predict(X_test)


# Algorithm 3: Multi Layer Perceptron
class MultiLayerPerceptron:
    def __init__(self, hidden_neurons, epochs, learning_rate=0.1, random_seed=42):
        self.hidden_neurons = hidden_neurons
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.name = (
            f"Multi-Layer Perceptron ({hidden_neurons} neurons, {epochs} epochs)"
        )

        self.model = MLPClassifier(
            hidden_layer_sizes=(hidden_neurons,),
            max_iter=epochs,
            learning_rate_init=learning_rate,
            random_state=random_seed,
            activation="logistic",  # Sigmoid activation
            solver="sgd",  # Stochastic Gradient Descent
            learning_rate="constant",  # Constant learning rate
            momentum=0,  # No momentum
            alpha=0,  # No L2 regularization
            batch_size="auto",  # Auto batch size
            verbose=False,  # No verbose output during training (no iteration loss printing)
        )

        # Add a scaler for feature normalization
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        print(f"\nTraining {self.name}...")
        # Normalize features before training
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        print("Training complete!")

    def predict(self, X_test):
        # Normalize test data using the same scaler
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

# Algorithm 4: Logistic Regression
class LogisticRegressionModel:
    def __init__(
        self,
        epochs,
        C=1.0,
        random_seed=42,
    ):
        self.C = C
        self.random_seed = random_seed
        self.model = LogisticRegression(C=C, random_state=random_seed, max_iter=epochs)
        self.name = f"Logistic Regression (C={C}, epochs={epochs})"
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        print(f"\nTraining {self.name}...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        print("Training complete!")

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)


# Algorithm 5: Random Forest
class RandomForestModel:
    def __init__(self, n_estimators, random_seed=42):
        self.n_estimators = n_estimators
        self.random_seed = random_seed
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_seed
        )
        self.name = f"Random Forest (n_estimators={n_estimators})"

    def train(self, X_train, y_train):
        print(f"\nTraining {self.name}...")
        self.model.fit(X_train, y_train)
        print("Training complete!")

    def predict(self, X_test):
        return self.model.predict(X_test)


# Algorithm 6: Linear Support Vector Classifier
class LinearSVCModel:
    def __init__(self, epochs, C=1.0, random_seed=42):
        self.C = C
        self.random_seed = random_seed
        self.model = LinearSVC(C=C, random_state=random_seed, max_iter=epochs)
        self.name = f"Linear SVC (C={C}, epochs={epochs})"
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        print(f"\nTraining {self.name}...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        print("Training complete!")

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

# Algorithm 7: Extra Trees
class ExtraTreesModel:
    def __init__(self, n_estimators, random_seed=42):
        self.n_estimators = n_estimators
        self.random_seed = random_seed
        self.model = ExtraTreesClassifier(
            n_estimators=n_estimators, random_state=random_seed
        )
        self.name = f"Extra Trees (n_estimators={n_estimators})"

    def train(self, X_train, y_train):
        print(f"\nTraining {self.name}...")
        self.model.fit(X_train, y_train)
        print("Training complete!")

    def predict(self, X_test):
        return self.model.predict(X_test)


# Algorithm 8: Gaussian Naive Bayes
class GaussianNBModel:
    def __init__(self):
        self.model = GaussianNB()
        self.name = "Gaussian Naive Bayes"

    def train(self, X_train, y_train):
        print(f"\nTraining {self.name}...")
        self.model.fit(X_train, y_train)
        print("Training complete!")

    def predict(self, X_test):
        return self.model.predict(X_test)


# Algorithm 9: Decision Tree
class DecisionTreeModel:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.model = DecisionTreeClassifier(random_state=random_seed)
        self.name = "Decision Tree"

    def train(self, X_train, y_train):
        print(f"\nTraining {self.name}...")
        self.model.fit(X_train, y_train)
        print("Training complete!")

    def predict(self, X_test):
        return self.model.predict(X_test)


# Algorithm 10: AdaBoost
class AdaBoostModel:
    def __init__(self, n_estimators, random_seed=42):
        self.n_estimators = n_estimators
        self.random_seed = random_seed
        self.model = AdaBoostClassifier(
            n_estimators=n_estimators, random_state=random_seed
        )
        self.name = f"AdaBoost (n_estimators={n_estimators})"

    def train(self, X_train, y_train):
        print(f"\nTraining {self.name}...")
        self.model.fit(X_train, y_train)
        print("Training complete!")

    def predict(self, X_test):
        return self.model.predict(X_test)


# Algorithm 11: Stochastic Gradient Descent
class SGDModel:
    def __init__(self, epochs, random_seed=42):
        self.random_seed = random_seed
        self.model = SGDClassifier(random_state=random_seed, max_iter=epochs, tol=1e-3)
        self.name = "Stochastic Gradient Descent (epochs={epochs})"
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        print(f"\nTraining {self.name}...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        print("Training complete!")

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)


# Algorithm 12: Passive Aggressive
class PassiveAggressiveModel:
    def __init__(self, epochs, random_seed=42):
        self.random_seed = random_seed
        self.model = PassiveAggressiveClassifier(
            random_state=random_seed, max_iter=epochs, tol=1e-3
        )
        self.name = "Passive Aggressive (epochs={epochs})"
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        print(f"\nTraining {self.name}...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        print("Training complete!")

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
