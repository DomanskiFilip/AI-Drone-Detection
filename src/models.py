import numpy
# Algorythm 1 Euclidean Distance: looking at nearest neighbour and predicting as that class
class EuclideanDistance:
  def __init__(self):
      self.X_train = None
      self.y_train = None
      self.name = "Euclidean Distance"

  def train(self, X_train, y_train):
        print(f"\nTraining {self.name}...")
        self.X_train = X_train
        self.y_train = y_train
        print(f"Memorized {len(X_train)} training examples")

  def predict_single(self, x):
    # Calculate Euclidean distance to all training samples
      # Distance formula: sqrt(sum((x - train_sample)^2))
      distances = numpy.sqrt(numpy.sum((self.X_train - x)**2, axis=1))

      # Find the index of the smallest distance
      closest_index = numpy.argmin(distances)

      # Return the label of the closest training sample
      return self.y_train[closest_index]

  def predict(self, X_test):
    predictions = []
    for x in X_test:
        predictions.append(self.predict_single(x))
    return numpy.array(predictions)

# Algorythm 2 K Nearest Neighbours: looking at 3 nearest neighbours and predicting as most popular class among them
class KNearestNeighbors:
  def __init__(self, k=3):
        # k: Number of neighbors to vote (default 3)
        self.k = k
        self.X_train = None
        self.y_train = None
        self.name = f"K-Nearest Neighbors (k={k})"

  def train(self, X_train, y_train):
      print(f"\nTraining {self.name}...")
      self.X_train = numpy.array(X_train)
      self.y_train = numpy.array(y_train)
      print(f"Memorized {len(X_train)} training examples")

  def predict_single(self, x):
      # Calculate distances
      distances = numpy.sqrt(numpy.sum((self.X_train - x)**2, axis=1))

      # Get indices of K nearest neighbors
      # numpy.argpartition finds K smallest values efficiently
      k_indices = numpy.argpartition(distances, self.k)[:self.k].astype(int)

      # Get labels of K nearest neighbors
      k_labels = self.y_train[k_indices]

      # Count votes for each class
      # Find which class got the most votes
      unique_labels, counts = numpy.unique(k_labels, return_counts=True)
      winner = unique_labels[numpy.argmax(counts)]

      return winner

  def predict(self, X_test):
      predictions = []
      for x in X_test:
          predictions.append(self.predict_single(x))
      return numpy.array(predictions)


# Algorythm 3 Multi Layer Perceptron: A calssical neural network that trains x ammount of perceprtons over y epochs then predicts based on activated trained features
class MultiLayerPerceptron:
  def __init__(self, hidden_neurons=100, epochs=30, learning_rate=0.1, random_seed=42):
      self.hidden_neurons = hidden_neurons
      self.epochs = epochs
      self.learning_rate = learning_rate
      self.random_seed = random_seed
      self.name = f"Multi-Layer Perceptron ({hidden_neurons} neurons, {epochs} epochs)"

      # These will be set during training
      self.weights_input_hidden = None
      self.bias_hidden = None
      self.weights_hidden_output = None
      self.bias_output = None
      self.feature_mean = None
      self.feature_std = None
      self.n_classes = None
      self.input_size = None

  def sigmoid(self, x):
      # Sigmoid activation function: squashes values to range (0, 1)
      # Formula: 1 / (1 + e^(-x))
      return 1.0 / (1.0 + numpy.exp(-numpy.clip(x, -500, 500)))

  def sigmoid_derivative(self, activated_value):
      # Derivative of sigmoid - used in backpropagation
      # Formula: sigmoid(x) * (1 - sigmoid(x))

      return activated_value * (1.0 - activated_value)

  def normalize_features(self, X):
      # Normalize features to have mean=0, std=1
      if self.feature_mean is None:
          # First time - calculate mean and std from training data
          self.feature_mean = numpy.mean(X, axis=0)
          self.feature_std = numpy.std(X, axis=0)
          # Prevent division by zero
          self.feature_std[self.feature_std < 1e-9] = 1.0

      # Apply normalization
      return (X - self.feature_mean) / self.feature_std

  def initialize_weights(self):
      # Initialize weights randomly
      numpy.random.seed(self.random_seed)
      weight_range = 0.1

      # Weights from input to hidden layer
      self.weights_input_hidden = (numpy.random.rand(self.hidden_neurons, self.input_size) * 2 - 1) * weight_range
      self.bias_hidden = (numpy.random.rand(self.hidden_neurons) * 2 - 1) * weight_range

      # Weights from hidden to output layer
      self.weights_hidden_output = (numpy.random.rand(self.n_classes, self.hidden_neurons) * 2 - 1) * weight_range
      self.bias_output = (numpy.random.rand(self.n_classes) * 2 - 1) * weight_range

  def forward_pass(self, x):
      """
      Forward pass: push data through the network

      Steps:
      1. Input -> Hidden layer (with sigmoid activation)
      2. Hidden -> Output layer (with sigmoid activation)
      """
      # Input to hidden layer
      hidden_sum = numpy.dot(self.weights_input_hidden, x) + self.bias_hidden
      hidden_activation = self.sigmoid(hidden_sum)

      # Hidden to output layer
      output_sum = numpy.dot(self.weights_hidden_output, hidden_activation) + self.bias_output
      output_activation = self.sigmoid(output_sum)

      return hidden_activation, output_activation

  def train(self, X_train, y_train):
      """
      Train the neural network using backpropagation

      Backpropagation:
      1. Make a prediction (forward pass)
      2. Calculate how wrong we were (error)
      3. Adjust weights to reduce error (backward pass)
      4. Repeat many times
      """
      print(f"\nTraining {self.name}...")
      print("This may take a minute...")

      # Get dimensions
      n_samples, self.input_size = X_train.shape
      self.n_classes = len(numpy.unique(y_train))

      # Normalize features
      X_normalized = self.normalize_features(X_train)

      # Initialize weights
      self.initialize_weights()

      # Training loop
      for epoch in range(self.epochs):
          correct = 0

          # Go through each training example
          for i in range(n_samples):
              x = X_normalized[i]
              target_class = y_train[i]

              # FORWARD PASS: Make a prediction
              hidden, output = self.forward_pass(x)

              # Create target vector (one-hot encoding)
              # Example: if target_class=1 and n_classes=3, target=[0, 1, 0]
              target = numpy.zeros(self.n_classes)
              target[target_class] = 1.0

              # CALCULATE ERROR
              # Output layer error
              output_error = target - output
              output_delta = output_error * self.sigmoid_derivative(output)

              # Hidden layer error (backpropagate)
              hidden_error = numpy.dot(self.weights_hidden_output.T, output_delta)
              hidden_delta = hidden_error * self.sigmoid_derivative(hidden)

              # UPDATE WEIGHTS (gradient descent)
              # Hidden to output weights
              self.weights_hidden_output += self.learning_rate * numpy.outer(output_delta, hidden)
              self.bias_output += self.learning_rate * output_delta

              # Input to hidden weights
              self.weights_input_hidden += self.learning_rate * numpy.outer(hidden_delta, x)
              self.bias_hidden += self.learning_rate * hidden_delta

              # Track accuracy
              if numpy.argmax(output) == target_class:
                  correct += 1

          # Print progress every 10 epochs
          if (epoch + 1) % 10 == 0:
              accuracy = (correct / n_samples) * 100
              print(f"  Epoch {epoch + 1}/{self.epochs} - Training accuracy: {accuracy:.2f}%")

      print("Training complete!")

  def predict_single(self, x):
      _, output = self.forward_pass(x)
      return numpy.argmax(output)

  def predict(self, X_test):
      # Normalize test data using training statistics
      X_normalized = (X_test - self.feature_mean) / self.feature_std

      predictions = []
      for x in X_normalized:
          predictions.append(self.predict_single(x))

      return numpy.array(predictions)

