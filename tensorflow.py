import tensorflow as tf

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# Compile the model (not shown in this snippet)
# Specify loss functions, optimizer, and other hyperparameters

# Example mock data for testing
sample_input = x_train[0:1]  # Use the first training example
predicted_logits = model(sample_input)

# Print the logits (raw scores) for each class
print("Logits for each class:")
print(predicted_logits.numpy())

# Note: In practice, you would compile the model and train it using real data.

# Feel free to explore more TensorFlow examples on GitHub or the official TensorFlow website!
