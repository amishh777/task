from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()

images = digits.images #assigns each pixel a value
correct_value = digits.target # conatins correct output associated w each img

# Convert each 8x8 image into 64 pixel values
X = images.reshape(images.shape[0], 64)

# Scale pixel values from 0-16 to 0-1
X = X / 16.0

# Keep the original digit labels: 0 through 9
y = correct_value.astype(int) # ensure that correct_value is an integer, more of a safety feature here, code would work regardless

for digit in range(0, 10, 1):
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

# Define the network size
input_size = X_train.shape[1]   # 64 pixels
hidden_size = 32                # 32 hidden neurons
output_size = 10                # digits 0 through 9

# Create random values for the weights associated with each neuron to pixel link
parameter_random = np.random.default_rng(123)

W1 = parameter_random.normal(0, np.sqrt(2.0 / input_size),(input_size, hidden_size))

b1 = np.zeros(hidden_size) # weights and biases for pixels to first layer of neurons

W2 = parameter_random.normal(0, np.sqrt(2.0 / hidden_size), (hidden_size, output_size))

b2 = np.zeros(output_size) # weights and biasees for first to final layer

def softmax(scores):
    probabilities = np.zeros_like(scores)

    for image_number in range(0, scores.shape[0], 1):
        current_scores = scores[image_number]

        biggest_score = np.max(current_scores)
        shifted_scores = current_scores - biggest_score

        exponential_scores = np.exp(shifted_scores)   
        # all shifted scores to e^shifted_scores

        total = np.sum(exponential_scores) 
        # sum of all exponential scores for this image

        probabilities[image_number] = exponential_scores / total
        #updates for each image_number the probabilities of each digit in the array

    return probabilities

#training settings
number_of_steps= 1000
learning rate=0.1

for step in range(0,number_of_steps,1):
        # Hidden-layer forward pass
    Z1 = X_train @ W1 + b1 #Matrix multiplication and addition of biases
    A1 = np.maximum(0, Z1) #Z1 after ReLU
    
    # Second-layer forward pass: produce one raw score for each digit
    scores = A1 @ W2 + b2 
    
    # Convert the 10 raw scores into 10 probabilities
    P = softmax(scores)
    
    # Create one-hot target rows for the correct digit of each image
    target_values = np.zeros_like(P)
    
    for image_number in range(0, y_train.shape[0], 1):
        correct_digit = y_train[image_number]
        target_values[image_number][correct_digit] = 1
    
    total_loss = 0
    
    for image_number in range(0, y_train.shape[0], 1):
        correct_digit = y_train[image_number]
        correct_probability = P[image_number][correct_digit]
        correct_probability = max(correct_probability, 1e-12)
        image_loss = -np.log(correct_probability)
        total_loss = total_loss + image_loss
    
    loss = total_loss / y_train.shape[0]
    
    # With softmax and cross-entropy, this is the gradient of the average loss
    # with respect to the raw output scores.
    d_scores = (P - target_values) / y_train.shape[0]
    
    # Gradients for the second layer
    dW2 = A1.T @ d_scores
    db2 = np.sum(d_scores, axis=0)
    # Move the gradient backward into the hidden layer
    dA1 = d_scores @ W2.T
    # ReLU derivative
    dZ1 = dA1 * (Z1 > 0)
    # Gradients for the first layer
    dW1 = X_train.T @ dZ1
    db1 = np.sum(dZ1, axis=0)
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
