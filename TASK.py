from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()

images = digits.images
correct_value = digits.target

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
y = correct_value.astype(int)

print("\nPrepared input shape:", X.shape)
print("Prepared answer shape:", y.shape)
print("Classes:", sorted(set(y)))

for digit in range(10):
    print(f"Number of digit {digit} images:", np.sum(y == digit))
     #THIS PIECE of code lines 1-30 loads the digits dataset, converts into a 64 pixel value array and labels each image with its correct output value
