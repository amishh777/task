from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()

images = digits.images #assigns each pixel a value
correct_value = digits.target # conatins correct output associated w each img

print("Image shape:", images.shape)
print("Correct-value shape:", correct_value.shape)
print("First correct value:", correct_value[0])
print("First image:")
print(images[0])

# Convert each 8x8 image into 64 pixel values
X = images.reshape(images.shape[0], 64)

# Scale pixel values from 0-16 to 0-1
X = X / 16.0

# Keep the original digit labels: 0 through 9
y = correct_value.astype(int) # ensure that correct_value is an integer, more of a safety feature here, code would work regardless

print("\nPrepared input shape:", X.shape)
print("Prepared answer shape:", y.shape)
print("Classes:", sorted(set(y))) # labels all different outputs in coorect_values

for digit in range(10):
    print(f"Number of digit {digit} images:", np.sum(y == digit))
     #THIS PIECE of code lines 1-30 loads the digits dataset, converts into a 64 pixel value array and labels each image with its correct output value

# Create a repeatable random shuffle of the dataset rows
random_seq = np.random.default_rng(42)
shuffled_index = random_seq.permutation(X.shape[0])

# Shuffle images and their matching answers in the same order
X = X[shuffled_index]
y = y[shuffled_index]

# Use 80% of the data for training and the remaining 20% for testing
split_index = int(0.8 * X.shape[0])

X_train = X[:split_index]
y_train = y[:split_index]

# X[1437:] means rows 1437 through the end.
# Since there are 1797 images, this is the last 360 images, approximately 20%.
X_test = X[split_index:]
y_test = y[split_index:]

print("\nTraining input shape:", X_train.shape)
print("Training answer shape:", y_train.shape)
print("Testing input shape:", X_test.shape)
print("Testing answer shape:", y_test.shape)

# Define the network size
input_size = X_train.shape[1]   # 64 pixels
hidden_size = 32                # 32 hidden neurons
output_size = 10                # digits 0 through 9

# Create random values for the weights
parameter_random = np.random.default_rng(123)

W1 = parameter_random.normal(
    0,
    np.sqrt(2.0 / input_size),
    (input_size, hidden_size)
)

b1 = np.zeros(hidden_size)

W2 = parameter_random.normal(
    0,
    np.sqrt(2.0 / hidden_size),
    (hidden_size, output_size)
)

b2 = np.zeros(output_size)

print("\nNetwork parameter shapes:")
print("W1:", W1.shape)
print("b1:", b1.shape)
print("W2:", W2.shape)
print("b2:", b2.shape)
