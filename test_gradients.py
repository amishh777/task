from sklearn.datasets import load_digits
import numpy as np
answer = input("Print accuracy on training iterations? (y/n): ").lower() 
digits = load_digits()

images = digits.images #assigns each pixel a value
correct_value = digits.target # conatins correct output associated w each img

# Convert each 8x8 image into 64 pixel values
pixel_digit = images.reshape(images.shape[0], 64)

# Scale pixel values from 0-16 to 0-1
pixel_digit = pixel_digit / 16.0

# Keep the original digit labels: 0 through 9
correct_digits = correct_value.astype(int) # ensure that correct_value is an integer, more of a safety feature here, code would work regardless

for digit in range(0, 10, 1):
    print(f"Number of digit {digit} images:", np.sum(correct_digits == digit))
     #THIS PIECE of code lines 1-30 loads the digits dataset, converts into a 64 pixel value array and labels each image with its correct output value

# Create a repeatable random shuffle of the dataset rows
random_seq = np.random.default_rng(42)
shuffled_index = random_seq.permutation(pixel_digit.shape[0])

# Shuffle images and their matching answers in the same order
pixel_digit = pixel_digit[shuffled_index]
correct_digits = correct_digits[shuffled_index]

# Use 80% of the data for training and the remaining 20% for testing
split_index = int(0.8 * pixel_digit.shape[0])

pixel_digit_train = pixel_digit[:split_index]
correct_digits_train = correct_digits[:split_index]

# pixel_digit[1437:] means rows 1437 through the end.
# Since there are 1797 images, this is the last 360 images, approximately 20%.
pixel_digit_test = pixel_digit[split_index:]
correct_digits_test = correct_digits[split_index:]

# Define the network size
input_size = pixel_digit_train.shape[1]   # 64 pixels
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

def calculate_loss(probabilities, correct_digits):
    total_loss = 0

    for image_number in range(0, correct_digits.shape[0], 1):
        correct_digit = correct_digits[image_number]

        correct_probability = probabilities[image_number][correct_digit]
        correct_probability = max(correct_probability, 1e-12)

        image_loss = -np.log(correct_probability)
        total_loss = total_loss + image_loss

    return total_loss / correct_digits.shape[0]


# Use a small batch for gradient checking
pixel_digit_check = pixel_digit_train[:20].copy()
correct_digits_check = correct_digits_train[:20].copy()

# Forward pass
Z1 = pixel_digit_check @ W1 + b1
A1 = np.maximum(0, Z1)

scores = A1 @ W2 + b2
P = softmax(scores)

# Create one-hot target values
target_values = np.zeros_like(P)

for image_number in range(0, correct_digits_check.shape[0], 1):
    correct_digit = correct_digits_check[image_number]
    target_values[image_number][correct_digit] = 1


# Manual gradient
d_scores = (P - target_values) / correct_digits_check.shape[0]
dW2 = A1.T @ d_scores


# Select one W2 weight
row = 0
column = 0
small_change = 0.00001

original_value = W2[row][column]


# Calculate loss after increasing the weight
W2[row][column] = original_value + small_change

scores_plus = A1 @ W2 + b2
P_plus = softmax(scores_plus)
loss_plus = calculate_loss(P_plus, correct_digits_check)


# Calculate loss after decreasing the weight
W2[row][column] = original_value - small_change

scores_minus = A1 @ W2 + b2
P_minus = softmax(scores_minus)
loss_minus = calculate_loss(P_minus, correct_digits_check)


# Restore the original weight
W2[row][column] = original_value


# Calculate the numerical gradient
numerical_gradient = (
    loss_plus - loss_minus
) / (2 * small_change)


# Get the manual gradient
manual_gradient = dW2[row][column]

print("Manual gradient:", manual_gradient)
print("Numerical gradient:", numerical_gradient)


if abs(numerical_gradient-manual_gradient < 0.0000001)
    print("Gradient check passed")
else:
    print("Gradient check failed")
